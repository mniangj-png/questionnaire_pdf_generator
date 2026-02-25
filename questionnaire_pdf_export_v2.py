
import ast
import io
import os
import re
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
)


@dataclass
class LabelRecord:
    section_key: str
    section_title_fr: str
    section_title_en: str
    function_name: str
    lineno: int
    widget: str
    slot: str
    fr: str
    en: str
    conditions: List[str]


# -------------------------
# AST helpers
# -------------------------

def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_st_call(call: ast.AST) -> bool:
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "st"
    )


def _t_pair_from_call(call: ast.AST) -> Optional[Tuple[str, str]]:
    if not isinstance(call, ast.Call):
        return None
    if not _is_name(call.func, "t"):
        return None
    if len(call.args) < 3:
        return None
    a_fr, a_en = call.args[1], call.args[2]
    if (
        isinstance(a_fr, ast.Constant)
        and isinstance(a_fr.value, str)
        and isinstance(a_en, ast.Constant)
        and isinstance(a_en.value, str)
    ):
        return a_fr.value, a_en.value
    return None


def _extract_t_calls(node: ast.AST) -> List[Tuple[ast.Call, str, str]]:
    out: List[Tuple[ast.Call, str, str]] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            pair = _t_pair_from_call(sub)
            if pair:
                out.append((sub, pair[0], pair[1]))
    out.sort(key=lambda x: (getattr(x[0], "lineno", 0), getattr(x[0], "col_offset", 0)))
    return out


def _find_functions(tree: ast.Module) -> Dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _extract_steps(funcs: Dict[str, ast.FunctionDef]) -> Dict[str, Dict[str, str]]:
    steps = {}
    fn = funcs.get("get_steps")
    if not fn:
        return steps
    for n in ast.walk(fn):
        if isinstance(n, ast.Tuple) and len(n.elts) == 2:
            k, v = n.elts
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                pair = _t_pair_from_call(v)
                if pair:
                    steps[k.value] = {"fr": pair[0], "en": pair[1]}
    return steps


def _func_to_section_key(func_name: str) -> Optional[str]:
    if func_name == "rubric_send":
        return "SEND"
    m = re.fullmatch(r"rubric_(\d+)", func_name)
    if m:
        return f"R{int(m.group(1))}"
    return None


class _UILabelVisitor(ast.NodeVisitor):
    def __init__(self, src: str, steps: Dict[str, Dict[str, str]]):
        self.src = src
        self.steps = steps
        self.records: List[LabelRecord] = []
        self.func_name: Optional[str] = None
        self.if_stack: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev = self.func_name
        self.func_name = node.name
        self.generic_visit(node)
        self.func_name = prev

    def visit_If(self, node: ast.If):
        cond = ast.unparse(node.test) if hasattr(ast, "unparse") else "<condition>"
        self.if_stack.append(cond)
        for s in node.body:
            self.visit(s)
        self.if_stack.pop()
        if node.orelse:
            self.if_stack.append(f"ELSE of ({cond})")
            for s in node.orelse:
                self.visit(s)
            self.if_stack.pop()

    def visit_Call(self, node: ast.Call):
        if _is_st_call(node) and self.func_name:
            section_key = _func_to_section_key(self.func_name)
            if section_key:
                titles = self.steps.get(section_key, {"fr": section_key, "en": section_key})
                widget = node.func.attr
                for i, arg in enumerate(node.args):
                    for _, fr, en in _extract_t_calls(arg):
                        self.records.append(
                            LabelRecord(
                                section_key=section_key,
                                section_title_fr=titles["fr"],
                                section_title_en=titles["en"],
                                function_name=self.func_name,
                                lineno=getattr(node, "lineno", 0),
                                widget=widget,
                                slot=f"arg{i}",
                                fr=fr,
                                en=en,
                                conditions=list(self.if_stack),
                            )
                        )
                for kw in node.keywords or []:
                    if kw.value is None:
                        continue
                    for _, fr, en in _extract_t_calls(kw.value):
                        self.records.append(
                            LabelRecord(
                                section_key=section_key,
                                section_title_fr=titles["fr"],
                                section_title_en=titles["en"],
                                function_name=self.func_name,
                                lineno=getattr(node, "lineno", 0),
                                widget=widget,
                                slot=(kw.arg or "kw"),
                                fr=fr,
                                en=en,
                                conditions=list(self.if_stack),
                            )
                        )
        self.generic_visit(node)


def _normalize_text_for_pdf(text: str) -> str:
    txt = text.replace("\r\n", "\n").replace("\r", "\n")
    txt = re.sub(r"^\s+|\s+$", "", txt)
    txt = re.sub(r"^[#]+\s*", "", txt, flags=re.MULTILINE)
    txt = txt.replace("**", "")
    txt = txt.replace("`", "")
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _extract_constants(src: str) -> Dict[str, Any]:
    tree = ast.parse(src)
    out: Dict[str, Any] = {}
    wanted = {"ROLE_OPTIONS_FR", "ROLE_OPTIONS_EN", "SCORE_SCALES"}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    try:
                        out[target.id] = ast.literal_eval(n.value)
                    except Exception:
                        pass
    return out


def extract_questionnaire_labels_from_source(app_py_path: str) -> Dict[str, Any]:
    with open(app_py_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    funcs = _find_functions(tree)
    steps = _extract_steps(funcs)

    visitor = _UILabelVisitor(src, steps)
    targets = [f"rubric_{i}" for i in range(1, 13)] + ["rubric_send"]
    for name in targets:
        fn = funcs.get(name)
        if fn:
            visitor.visit(fn)

    seen = set()
    deduped: List[LabelRecord] = []
    order = list(steps.keys())
    order_idx = {k: i for i, k in enumerate(order)}
    for r in sorted(
        visitor.records,
        key=lambda x: (
            order_idx.get(x.section_key, 999),
            x.lineno,
            x.widget,
            x.slot,
            x.fr,
            x.en,
        ),
    ):
        k = (r.section_key, r.lineno, r.widget, r.slot, r.fr, r.en)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)

    return {
        "app_file": os.path.basename(app_py_path),
        "app_path": os.path.abspath(app_py_path),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps": steps,
        "records": [asdict(r) for r in deduped],
        "constants": _extract_constants(src),
        "notes": {
            "scope": "Extraction AST des appels t(lang, fr, en) présents dans rubric_1..rubric_12 et rubric_send.",
            "limits": [
                "Les libellés dynamiques provenant des fichiers de données (longlist, pays) ne sont pas tous visibles via l’AST.",
                "Les options construites à l’exécution peuvent nécessiter une annexe complémentaire."
            ],
        },
    }


# -------------------------
# Longlist loader (CSV/XLSX)
# -------------------------

def _normalize_longlist_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame normalisé avec :
    domain_code, domain_label_fr, domain_label_en, stat_code, stat_label_fr, stat_label_en
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "domain_code", "domain_label_fr", "domain_label_en",
            "stat_code", "stat_label_fr", "stat_label_en"
        ])

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Déjà au bon format (CSV normalisé)
    normalized_cols = {
        "domain_code", "domain_label_fr", "domain_label_en",
        "stat_code", "stat_label_fr", "stat_label_en"
    }
    if normalized_cols.issubset(set(df.columns)):
        out = df[list(normalized_cols)].copy()
        for c in out.columns:
            out[c] = out[c].astype(str).fillna("").str.strip()
        out = out[[
            "domain_code", "domain_label_fr", "domain_label_en",
            "stat_code", "stat_label_fr", "stat_label_en"
        ]]
        return out

    # Format xlsx "longlist.xlsx"
    if {"Domain_code", "Domain_label_fr", "Stat_label_fr"}.issubset(set(df.columns)):
        out = pd.DataFrame()
        out["domain_code"] = df["Domain_code"].astype(str).fillna("").str.strip()
        out["domain_label_fr"] = (
            df["Domain_label_fr"].astype(str).fillna("").str.split("|", n=1).str[-1].str.strip()
        )
        out["stat_code"] = (
            df["Stat_label_fr"].astype(str).fillna("").str.split("|", n=1).str[0].str.strip()
        )
        out["stat_label_fr"] = (
            df["Stat_label_fr"].astype(str).fillna("").str.split("|", n=1).str[-1].str.strip()
        )
        if "Domain_label_en" in df.columns:
            out["domain_label_en"] = (
                df["Domain_label_en"].astype(str).fillna("").str.split("|", n=1).str[-1].str.strip()
            )
        else:
            out["domain_label_en"] = out["domain_label_fr"]

        if "Stat_label_en" in df.columns:
            out["stat_label_en"] = (
                df["Stat_label_en"].astype(str).fillna("").str.split("|", n=1).str[-1].str.strip()
            )
        else:
            out["stat_label_en"] = out["stat_label_fr"]

        return out[[
            "domain_code", "domain_label_fr", "domain_label_en",
            "stat_code", "stat_label_fr", "stat_label_en"
        ]]

    # Mappages tolérants (selon variantes de colonnes)
    aliases = {
        "domain_code": ["domain_code", "Domain_code", "domain", "Domain"],
        "domain_label_fr": ["domain_label_fr", "Domain_label_fr", "domaine_fr", "domain_fr"],
        "domain_label_en": ["domain_label_en", "Domain_label_en", "domaine_en", "domain_en"],
        "stat_code": ["stat_code", "Stat_code", "indicator_code", "indicator_id"],
        "stat_label_fr": ["stat_label_fr", "Stat_label_fr", "indicator_label_fr", "label_fr"],
        "stat_label_en": ["stat_label_en", "Stat_label_en", "indicator_label_en", "label_en"],
    }
    picked = {}
    cols_set = set(df.columns)
    for target, cand in aliases.items():
        picked[target] = next((c for c in cand if c in cols_set), None)

    if picked["domain_code"] and picked["stat_label_fr"]:
        out = pd.DataFrame()
        out["domain_code"] = df[picked["domain_code"]].astype(str).fillna("").str.strip()
        out["domain_label_fr"] = (
            df[picked["domain_label_fr"]].astype(str).fillna("").str.strip()
            if picked["domain_label_fr"] else out["domain_code"]
        )
        out["domain_label_en"] = (
            df[picked["domain_label_en"]].astype(str).fillna("").str.strip()
            if picked["domain_label_en"] else out["domain_label_fr"]
        )
        out["stat_code"] = (
            df[picked["stat_code"]].astype(str).fillna("").str.strip()
            if picked["stat_code"] else ""
        )
        out["stat_label_fr"] = df[picked["stat_label_fr"]].astype(str).fillna("").str.strip()
        out["stat_label_en"] = (
            df[picked["stat_label_en"]].astype(str).fillna("").str.strip()
            if picked["stat_label_en"] else out["stat_label_fr"]
        )
        return out[[
            "domain_code", "domain_label_fr", "domain_label_en",
            "stat_code", "stat_label_fr", "stat_label_en"
        ]]

    return pd.DataFrame(columns=[
        "domain_code", "domain_label_fr", "domain_label_en",
        "stat_code", "stat_label_fr", "stat_label_en"
    ])


def load_longlist_for_annex(app_py_path: str) -> Dict[str, Any]:
    """
    Recherche la longlist dans l'ordre :
    - data/indicator_longlist.csv
    - indicator_longlist.csv
    - data/longlist.xlsx
    - longlist.xlsx
    dans le dossier de l'app puis le cwd.
    """
    app_dir = os.path.dirname(os.path.abspath(app_py_path))
    candidates = [
        os.path.join(app_dir, "data", "indicator_longlist.csv"),
        os.path.join(app_dir, "indicator_longlist.csv"),
        os.path.join(app_dir, "data", "longlist.xlsx"),
        os.path.join(app_dir, "longlist.xlsx"),
        os.path.join("data", "indicator_longlist.csv"),
        "indicator_longlist.csv",
        os.path.join("data", "longlist.xlsx"),
        "longlist.xlsx",
    ]

    found_path = None
    raw_df = None
    err = None

    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            if p.lower().endswith(".csv"):
                raw_df = pd.read_csv(p, dtype=str).fillna("")
            else:
                raw_df = pd.read_excel(p, dtype=str).fillna("")
            found_path = os.path.abspath(p)
            break
        except Exception as e:
            err = str(e)

    if raw_df is None:
        return {
            "found": False,
            "source_path": None,
            "error": err,
            "row_count": 0,
            "domain_count": 0,
            "en_missing_ratio": None,
            "df": pd.DataFrame(columns=[
                "domain_code", "domain_label_fr", "domain_label_en",
                "stat_code", "stat_label_fr", "stat_label_en"
            ]),
        }

    df = _normalize_longlist_df(raw_df)
    if df.empty:
        return {
            "found": True,
            "source_path": found_path,
            "error": "Format de longlist non reconnu",
            "row_count": 0,
            "domain_count": 0,
            "en_missing_ratio": None,
            "df": df,
        }

    # Nettoyage / tri
    for c in ["domain_code", "domain_label_fr", "domain_label_en", "stat_code", "stat_label_fr", "stat_label_en"]:
        df[c] = df[c].astype(str).fillna("").str.strip()

    # Fallback EN si vide
    df.loc[df["domain_label_en"] == "", "domain_label_en"] = df["domain_label_fr"]
    df.loc[df["stat_label_en"] == "", "stat_label_en"] = df["stat_label_fr"]

    df = df.drop_duplicates(subset=["domain_code", "stat_code", "stat_label_fr", "stat_label_en"]).copy()
    df = df.sort_values(
        by=["domain_code", "stat_code", "stat_label_fr", "stat_label_en"],
        kind="stable"
    ).reset_index(drop=True)

    en_missing_ratio = None
    if "stat_label_en" in df.columns and len(df) > 0:
        en_missing_ratio = float((df["stat_label_en"].astype(str).str.strip() == "").mean())

    return {
        "found": True,
        "source_path": found_path,
        "error": None,
        "row_count": int(len(df)),
        "domain_count": int(df["domain_code"].nunique(dropna=True)),
        "en_missing_ratio": en_missing_ratio,
        "df": df,
    }


# -------------------------
# PDF helpers / styles
# -------------------------

def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="TitleCenter", parent=s["Title"], alignment=TA_CENTER, leading=20))
    s.add(ParagraphStyle(name="Small", parent=s["BodyText"], fontSize=8.5, leading=10.5))
    s.add(ParagraphStyle(name="Tiny", parent=s["BodyText"], fontSize=7.3, leading=8.8))
    s.add(ParagraphStyle(name="CodeSmall", parent=s["BodyText"], fontName="Courier", fontSize=7.5, leading=9))
    s.add(ParagraphStyle(name="SectionH", parent=s["Heading2"], spaceBefore=10, spaceAfter=6))
    s.add(ParagraphStyle(name="AnnexH3", parent=s["Heading3"], fontSize=10.5, leading=12, spaceBefore=8, spaceAfter=4))
    return s


def _on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.7 * cm, f"Page {doc.page}")
    canvas.restoreState()


class OutlineDocTemplate(BaseDocTemplate):
    """
    BaseDocTemplate avec support des signets PDF (outline) via attributs sur les flowables :
    - _bookmark_name
    - _outline_title
    - _outline_level
    """
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_on_page)])
        self._outline_keys_seen = set()

    def afterFlowable(self, flowable):
        try:
            name = getattr(flowable, "_bookmark_name", None)
            title = getattr(flowable, "_outline_title", None)
            level = getattr(flowable, "_outline_level", None)
            if name and title is not None and level is not None:
                # destination sur la page courante
                self.canv.bookmarkPage(name)
                key = str(name)
                if key not in self._outline_keys_seen:
                    self.canv.addOutlineEntry(str(title), key, level=int(level), closed=False)
                    self._outline_keys_seen.add(key)
        except Exception:
            pass


def _mk_bookmark_paragraph(text: str, style, bookmark_name: str, outline_title: Optional[str] = None, outline_level: Optional[int] = None):
    p = Paragraph(escape(text), style)
    p._bookmark_name = bookmark_name
    p._outline_title = outline_title if outline_title is not None else text
    p._outline_level = outline_level if outline_level is not None else 0
    return p


def _internal_link_paragraph(text: str, bookmark_name: str, style):
    # ReportLab Paragraph supports internal links with href="#name"
    safe_text = escape(text)
    return Paragraph(f'<link href="#{bookmark_name}">{safe_text}</link>', style)


def _linkable_summary_table(rows: List[List[Any]], colWidths):
    t = Table(rows, colWidths=colWidths, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))
    return t


def _sanitize_bookmark_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")[:120] or "bk"


def _section_anchor(sec: str) -> str:
    return f"sec_{_sanitize_bookmark_key(sec)}"


def _annex_domain_anchor(domain_code: str) -> str:
    return f"annex_dom_{_sanitize_bookmark_key(domain_code)}"


# -------------------------
# PDF generation
# -------------------------

def build_questionnaire_pdf_bytes(
    extracted: Dict[str, Any],
    lang: str = "fr",
    version: str = "respondent",
    longlist_info: Optional[Dict[str, Any]] = None,
) -> bytes:
    assert lang in ("fr", "en")
    assert version in ("respondent", "technical")

    styles = _styles()
    buf = io.BytesIO()
    doc = OutlineDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.3 * cm,
        title="Questionnaire PDF export",
        author="Streamlit app22 PDF exporter",
    )
    story: List[Any] = []

    T = {
        "fr": {
            "title_resp": "Questionnaire app22 - version repondant",
            "title_tech": "Questionnaire app22 - version technique",
            "meta": "Metadonnees",
            "scope": "Perimetre de l'extraction",
            "limits": "Limites / points a verifier",
            "summary": "Sommaire cliquable",
            "section": "Section",
            "count": "Nombre d'entrees",
            "widget": "Widget",
            "slot": "Parametre",
            "line": "Ligne source",
            "conditions": "Conditions",
            "none": "Aucune",
            "annex": "Annexe - options statiques detectees",
            "annex_longlist": "Annexe - longlist complete FR / EN",
            "role_options": "Modalites de fonction (Rubrique 2)",
            "score_scales": "Baremes de notation (scoring)",
            "note_dynamic": "Note : les libelles dynamiques proviennent de fichiers de donnees externes. Cette v2 ajoute une annexe longlist FR/EN lorsque le fichier est disponible.",
            "generated": "Genere le (UTC)",
            "source": "Fichier source",
            "language": "Langue",
            "version": "Version",
            "longlist_source": "Source longlist",
            "longlist_rows": "Nombre de lignes longlist",
            "longlist_domains": "Nombre de domaines",
            "longlist_status_missing": "Longlist non trouvee (aucun fichier data/indicator_longlist.csv ni data/longlist.xlsx detecte).",
            "longlist_status_found": "Longlist chargee et annexe ajoutee.",
            "fr_label": "Libelle FR",
            "en_label": "Libelle EN",
            "code": "Code",
            "domain": "Domaine",
            "indicator": "Indicateur",
            "back_to_toc": "Retour au sommaire",
            "annex_nav": "Navigation annexe (domaines)",
            "send_section": "Envoi / finalisation",
            "source_file": "Fichier",
        },
        "en": {
            "title_resp": "app22 questionnaire - respondent version",
            "title_tech": "app22 questionnaire - technical version",
            "meta": "Metadata",
            "scope": "Extraction scope",
            "limits": "Limits / checks",
            "summary": "Clickable table of contents",
            "section": "Section",
            "count": "Number of entries",
            "widget": "Widget",
            "slot": "Parameter",
            "line": "Source line",
            "conditions": "Conditions",
            "none": "None",
            "annex": "Annex - detected static options",
            "annex_longlist": "Annex - complete longlist FR / EN",
            "role_options": "Role options (Section 2)",
            "score_scales": "Scoring scales",
            "note_dynamic": "Note: dynamic labels come from external data files. This v2 adds a FR/EN longlist annex when the file is available.",
            "generated": "Generated at (UTC)",
            "source": "Source file",
            "language": "Language",
            "version": "Version",
            "longlist_source": "Longlist source",
            "longlist_rows": "Longlist row count",
            "longlist_domains": "Domain count",
            "longlist_status_missing": "Longlist not found (no data/indicator_longlist.csv or data/longlist.xlsx detected).",
            "longlist_status_found": "Longlist loaded and annex added.",
            "fr_label": "FR label",
            "en_label": "EN label",
            "code": "Code",
            "domain": "Domain",
            "indicator": "Indicator",
            "back_to_toc": "Back to table of contents",
            "annex_nav": "Annex navigation (domains)",
            "send_section": "Submit / finalization",
            "source_file": "File",
        },
    }[lang]

    # Top title and metadata (bookmarked)
    title = T["title_resp"] if version == "respondent" else T["title_tech"]
    story.append(_mk_bookmark_paragraph(title, styles["TitleCenter"], "top", title, 0))
    story.append(Spacer(1, 0.25 * cm))

    meta_rows = [
        [T["source"], escape(str(extracted.get("app_file", "")))],
        [T["generated"], escape(str(extracted.get("generated_at_utc", "")))],
        [T["language"], "Français" if lang == "fr" else "English"],
        [T["version"], "Repondant" if (lang == "fr" and version == "respondent") else ("Technique" if lang == "fr" else ("Respondent" if version == "respondent" else "Technical"))],
    ]
    if longlist_info is not None:
        meta_rows.append([T["longlist_source"], escape(str(longlist_info.get("source_path") or "")) or "-"])
        meta_rows.append([T["longlist_rows"], str(longlist_info.get("row_count") or 0)])
        meta_rows.append([T["longlist_domains"], str(longlist_info.get("domain_count") or 0)])

    tbl = Table(meta_rows, colWidths=[5.0 * cm, 10.7 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.25 * cm))

    notes = extracted.get("notes", {}) or {}
    story.append(_mk_bookmark_paragraph(T["scope"], styles["SectionH"], "scope", T["scope"], 1))
    story.append(Paragraph(escape(str(notes.get("scope", ""))), styles["BodyText"]))
    story.append(Spacer(1, 0.12 * cm))
    story.append(_mk_bookmark_paragraph(T["limits"], styles["SectionH"], "limits", T["limits"], 1))
    for lim in notes.get("limits", []):
        story.append(Paragraph("• " + escape(str(lim)), styles["BodyText"]))
    story.append(Paragraph("• " + escape(T["note_dynamic"]), styles["BodyText"]))
    if longlist_info is not None:
        msg = T["longlist_status_found"] if longlist_info.get("found") and longlist_info.get("row_count", 0) > 0 else T["longlist_status_missing"]
        story.append(Paragraph("• " + escape(msg), styles["BodyText"]))
        if longlist_info.get("error"):
            story.append(Paragraph("• " + escape(f"Longlist error: {longlist_info['error']}"), styles["Small"]))
    story.append(Spacer(1, 0.25 * cm))

    records = extracted.get("records", []) or []
    steps = extracted.get("steps", {}) or {}
    order = list(steps.keys())
    order_index = {k: i for i, k in enumerate(order)}
    records.sort(
        key=lambda r: (
            order_index.get(r.get("section_key"), 999),
            int(r.get("lineno", 0)),
            str(r.get("widget", "")),
            str(r.get("slot", "")),
        )
    )

    # Summary / TOC table with internal links
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["section_key"]] = counts.get(r["section_key"], 0) + 1

    story.append(_mk_bookmark_paragraph(T["summary"], styles["SectionH"], "toc", T["summary"], 1))
    sum_rows: List[List[Any]] = [[T["section"], T["count"]]]
    for k in order:
        sec_title = steps[k][lang]
        if k == "SEND":
            sec_title = steps[k][lang] or T["send_section"]
        anchor = _section_anchor(k)
        sum_rows.append([_internal_link_paragraph(sec_title, anchor, styles["BodyText"]), str(counts.get(k, 0))])

    if longlist_info is not None:
        sum_rows.append([_internal_link_paragraph(T["annex_longlist"], "annex_longlist", styles["BodyText"]),
                         str(longlist_info.get("row_count") or 0)])

    story.append(_linkable_summary_table(sum_rows, colWidths=[13.0 * cm, 2.7 * cm]))
    story.append(PageBreak())

    # Group by section
    grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in order}
    for r in records:
        grouped.setdefault(r["section_key"], []).append(r)

    for sec in [k for k in order if k in grouped and grouped[k]]:
        sec_title = steps.get(sec, {"fr": sec, "en": sec})[lang]
        story.append(_mk_bookmark_paragraph(sec_title, styles["Heading1"], _section_anchor(sec), sec_title, 1))
        story.append(_internal_link_paragraph(T["back_to_toc"], "toc", styles["Small"]))
        story.append(Spacer(1, 0.10 * cm))

        sec_records = grouped[sec]
        if version == "respondent":
            seen_txt = set()
            idx = 0
            for r in sec_records:
                txt = _normalize_text_for_pdf(str(r[lang]))
                if not txt:
                    continue
                if txt in seen_txt:
                    continue
                seen_txt.add(txt)
                idx += 1
                parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
                prefix = f"{idx}. "
                if len(parts) == 1:
                    story.append(Paragraph(escape(prefix + parts[0]), styles["BodyText"]))
                else:
                    story.append(Paragraph(escape(prefix + parts[0]), styles["BodyText"]))
                    for p in parts[1:]:
                        story.append(Paragraph(escape(p), styles["BodyText"]))
                story.append(Spacer(1, 0.08 * cm))
        else:
            for i, r in enumerate(sec_records, start=1):
                txt = _normalize_text_for_pdf(str(r[lang]))
                header = f"{i}. [{T['line']} {r['lineno']}] {r['widget']} • {r['slot']}"
                story.append(Paragraph(escape(header), styles["Small"]))
                cond_txt = " ; ".join(str(c) for c in r.get("conditions", [])) if r.get("conditions") else T["none"]
                story.append(Paragraph(escape(f"{T['conditions']} : {cond_txt}"), styles["Small"]))
                if txt:
                    for p in [p.strip() for p in txt.split("\n\n") if p.strip()]:
                        story.append(Paragraph(escape(p), styles["BodyText"]))
                story.append(Spacer(1, 0.12 * cm))

        story.append(Spacer(1, 0.18 * cm))
        if sec != [k for k in order if k in grouped and grouped[k]][-1]:
            story.append(PageBreak())

    # Static constants annex
    constants = extracted.get("constants", {}) or {}
    if constants:
        story.append(PageBreak())
        story.append(_mk_bookmark_paragraph(T["annex"], styles["Heading1"], "annex_static", T["annex"], 1))
        story.append(_internal_link_paragraph(T["back_to_toc"], "toc", styles["Small"]))
        story.append(Spacer(1, 0.10 * cm))

        if "ROLE_OPTIONS_FR" in constants and "ROLE_OPTIONS_EN" in constants:
            story.append(_mk_bookmark_paragraph(T["role_options"], styles["SectionH"], "annex_roles", T["role_options"], 2))
            opts_fr = list(constants.get("ROLE_OPTIONS_FR") or [])
            opts_en = list(constants.get("ROLE_OPTIONS_EN") or [])
            max_len = max(len(opts_fr), len(opts_en))
            rows = [[T["fr_label"], T["en_label"]]]
            for i in range(max_len):
                rows.append([
                    escape(str(opts_fr[i])) if i < len(opts_fr) else "",
                    escape(str(opts_en[i])) if i < len(opts_en) else "",
                ])
            tt = Table(rows, colWidths=[7.85 * cm, 7.85 * cm], repeatRows=1)
            tt.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story.append(tt)
            story.append(Spacer(1, 0.12 * cm))

        if "SCORE_SCALES" in constants and isinstance(constants["SCORE_SCALES"], dict):
            story.append(_mk_bookmark_paragraph(T["score_scales"], styles["SectionH"], "annex_scales", T["score_scales"], 2))
            scales = constants["SCORE_SCALES"]
            for crit, d in scales.items():
                story.append(Paragraph(escape(str(crit)), styles["BodyText"]))
                mapping = (d.get(lang) or {}) if isinstance(d, dict) else {}
                for k_map, v_map in sorted(mapping.items(), key=lambda kv: int(kv[0])):
                    story.append(Paragraph("- " + escape(f"{k_map} = {v_map}"), styles["Small"]))
                story.append(Spacer(1, 0.08 * cm))

    # Longlist annex v2 (FR/EN)
    if longlist_info is not None:
        story.append(PageBreak())
        story.append(_mk_bookmark_paragraph(T["annex_longlist"], styles["Heading1"], "annex_longlist", T["annex_longlist"], 1))
        story.append(_internal_link_paragraph(T["back_to_toc"], "toc", styles["Small"]))
        story.append(Spacer(1, 0.10 * cm))

        ll_rows = [
            [T["source_file"], escape(str(longlist_info.get("source_path") or "-"))],
            [T["longlist_rows"], str(longlist_info.get("row_count") or 0)],
            [T["longlist_domains"], str(longlist_info.get("domain_count") or 0)],
        ]
        ll_tbl = Table(ll_rows, colWidths=[5.0 * cm, 10.7 * cm])
        ll_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ]))
        story.append(ll_tbl)
        story.append(Spacer(1, 0.15 * cm))

        df_long = longlist_info.get("df")
        if isinstance(df_long, pd.DataFrame) and not df_long.empty:
            # navigation list by domain (clickable internal links)
            story.append(_mk_bookmark_paragraph(T["annex_nav"], styles["SectionH"], "annex_longlist_nav", T["annex_nav"], 2))
            nav_rows = [[T["code"], T["domain"], "N"]]
            grouped_domains = (
                df_long.groupby(["domain_code", "domain_label_fr", "domain_label_en"], dropna=False)
                .size().reset_index(name="n")
                .sort_values(["domain_code", "domain_label_fr"], kind="stable")
            )
            for _, row in grouped_domains.iterrows():
                dc = str(row["domain_code"])
                dfr = str(row["domain_label_fr"])
                den = str(row["domain_label_en"])
                dlabel = f"{dc} - {dfr} / {den}"
                nav_rows.append([
                    _internal_link_paragraph(dc, _annex_domain_anchor(dc), styles["Small"]),
                    _internal_link_paragraph(dlabel, _annex_domain_anchor(dc), styles["Small"]),
                    str(int(row["n"])),
                ])
            nav_tbl = Table(nav_rows, colWidths=[2.1 * cm, 12.2 * cm, 1.4 * cm], repeatRows=1)
            nav_tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 7.7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (2, 1), (2, -1), "CENTER"),
            ]))
            story.append(nav_tbl)
            story.append(PageBreak())

            # Detailed listing by domain
            for idx_dom, (_, row) in enumerate(grouped_domains.iterrows(), start=1):
                dc = str(row["domain_code"])
                dfr = str(row["domain_label_fr"])
                den = str(row["domain_label_en"])
                title_dom = f"{dc} - {dfr} / {den}"
                story.append(_mk_bookmark_paragraph(title_dom, styles["AnnexH3"], _annex_domain_anchor(dc), title_dom, 2))
                story.append(_internal_link_paragraph(T["back_to_toc"], "annex_longlist_nav", styles["Tiny"]))
                story.append(Spacer(1, 0.05 * cm))

                sub = df_long[df_long["domain_code"].astype(str) == dc].copy()
                # compact table in portrait, FR and EN in separate rows within cells
                table_rows = [[T["code"], f"{T['indicator']} ({T['fr_label']} / {T['en_label']})"]]
                for _, r in sub.iterrows():
                    stat_code = str(r.get("stat_code", "") or "")
                    fr_label = _normalize_text_for_pdf(str(r.get("stat_label_fr", "") or ""))
                    en_label = _normalize_text_for_pdf(str(r.get("stat_label_en", "") or ""))
                    joined = (
                        f"<b>FR :</b> {escape(fr_label)}<br/>"
                        f"<b>EN :</b> {escape(en_label)}"
                    )
                    table_rows.append([
                        escape(stat_code),
                        Paragraph(joined, styles["Tiny"])
                    ])

                t_dom = Table(table_rows, colWidths=[2.4 * cm, 12.0 * cm], repeatRows=1)
                t_dom.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (0, -1), 7.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                story.append(t_dom)
                if idx_dom < len(grouped_domains):
                    story.append(PageBreak())
        else:
            story.append(Paragraph(escape(T["longlist_status_missing"]), styles["BodyText"]))
            if longlist_info.get("error"):
                story.append(Paragraph(escape(f"Longlist error: {longlist_info['error']}"), styles["Small"]))

    doc.build(story)
    return buf.getvalue()


def build_all_questionnaire_pdfs(app_py_path: str) -> Dict[str, bytes]:
    extracted = extract_questionnaire_labels_from_source(app_py_path)
    longlist_info = load_longlist_for_annex(app_py_path)

    outputs: Dict[str, bytes] = {}
    outputs["questionnaire_app22_repondant_FR.pdf"] = build_questionnaire_pdf_bytes(
        extracted, lang="fr", version="respondent", longlist_info=longlist_info
    )
    outputs["questionnaire_app22_technique_FR.pdf"] = build_questionnaire_pdf_bytes(
        extracted, lang="fr", version="technical", longlist_info=longlist_info
    )
    outputs["questionnaire_app22_respondent_EN.pdf"] = build_questionnaire_pdf_bytes(
        extracted, lang="en", version="respondent", longlist_info=longlist_info
    )
    outputs["questionnaire_app22_technical_EN.pdf"] = build_questionnaire_pdf_bytes(
        extracted, lang="en", version="technical", longlist_info=longlist_info
    )

    manifest = {
        "generated_at_utc": extracted.get("generated_at_utc"),
        "app_file": extracted.get("app_file"),
        "record_count": len(extracted.get("records", [])),
        "longlist": {
            "found": bool(longlist_info.get("found")),
            "source_path": longlist_info.get("source_path"),
            "row_count": longlist_info.get("row_count"),
            "domain_count": longlist_info.get("domain_count"),
            "error": longlist_info.get("error"),
        },
        "files": sorted(outputs.keys()),
        "note": "v2 includes clickable TOC (internal links + PDF outline bookmarks) and FR/EN longlist annex when available."
    }
    outputs["manifest_questionnaire_pdf_export_v2.json"] = __import__("json").dumps(
        manifest, ensure_ascii=False, indent=2
    ).encode("utf-8")
    return outputs


def build_all_questionnaire_pdfs_zip(app_py_path: str) -> bytes:
    files = build_all_questionnaire_pdfs(app_py_path)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    zbuf.seek(0)
    return zbuf.getvalue()


def render_questionnaire_pdf_export_panel(lang: str, app_py_path: Optional[str] = None) -> None:
    """
    Panneau Streamlit a inserer dans le dashboard Superadmin.
    v2 : annexes longlist FR/EN + sommaire cliquable / signets PDF.
    """
    if st is None:
        raise RuntimeError("Streamlit non disponible pour le panneau d'export PDF.")

    app_py_path = app_py_path or os.path.abspath(__file__)
    key_prefix = "qpdf_export_v2"

    def _tt(fr: str, en: str) -> str:
        return fr if lang == "fr" else en

    st.markdown("### " + _tt(
        "Questionnaire PDF v2 (FR/EN, repondant/technique)",
        "Questionnaire PDF v2 (FR/EN, respondent/technical)"
    ))
    st.caption(_tt(
        "Generation automatique depuis les libelles t(lang, fr, en) du code app22 + annexe longlist complete FR/EN (si fichier present) + signets PDF.",
        "Automatic generation from app22 t(lang, fr, en) labels + full FR/EN longlist annex (if file exists) + PDF bookmarks."
    ))

    longlist_info = load_longlist_for_annex(app_py_path)
    if longlist_info.get("found"):
        st.info(_tt(
            f"Longlist detectee : {longlist_info.get('row_count',0)} lignes, {longlist_info.get('domain_count',0)} domaines. Source : {longlist_info.get('source_path')}",
            f"Longlist detected: {longlist_info.get('row_count',0)} rows, {longlist_info.get('domain_count',0)} domains. Source: {longlist_info.get('source_path')}"
        ))
        if longlist_info.get("error"):
            st.warning(f"Longlist: {longlist_info['error']}")
    else:
        st.warning(_tt(
            "Longlist non detectee. L'annexe longlist sera vide. Ajoutez data/indicator_longlist.csv ou data/longlist.xlsx dans le repo GitHub.",
            "Longlist not detected. The longlist annex will be empty. Add data/indicator_longlist.csv or data/longlist.xlsx to the GitHub repo."
        ))

    c0, c1 = st.columns([1, 1])
    with c0:
        if st.button(_tt("Preparer les PDF v2", "Prepare v2 PDFs"), key=f"{key_prefix}_build"):
            try:
                extracted = extract_questionnaire_labels_from_source(app_py_path)
                ll = load_longlist_for_annex(app_py_path)
                bundle = {
                    "fr_resp": build_questionnaire_pdf_bytes(extracted, lang="fr", version="respondent", longlist_info=ll),
                    "fr_tech": build_questionnaire_pdf_bytes(extracted, lang="fr", version="technical", longlist_info=ll),
                    "en_resp": build_questionnaire_pdf_bytes(extracted, lang="en", version="respondent", longlist_info=ll),
                    "en_tech": build_questionnaire_pdf_bytes(extracted, lang="en", version="technical", longlist_info=ll),
                    "zip": build_all_questionnaire_pdfs_zip(app_py_path),
                    "record_count": len(extracted.get("records", [])),
                    "generated_at_utc": extracted.get("generated_at_utc"),
                    "longlist_rows": ll.get("row_count", 0),
                    "longlist_domains": ll.get("domain_count", 0),
                    "longlist_source": ll.get("source_path"),
                }
                st.session_state[f"{key_prefix}_bundle"] = bundle
                st.success(_tt(
                    "PDF v2 generes. Utilisez les boutons de telechargement ci-dessous.",
                    "v2 PDFs generated. Use the download buttons below."
                ))
            except Exception as e:
                st.error(f"PDF export v2 : {e}")

    with c1:
        st.info(_tt(
            "Important : placez la longlist dans le depot GitHub (data/indicator_longlist.csv ou data/longlist.xlsx). L'app Streamlit lira ces fichiers localement (sans Render ni Supabase).",
            "Important: place the longlist in the GitHub repo (data/indicator_longlist.csv or data/longlist.xlsx). The Streamlit app reads local files only (no Render, no Supabase)."
        ))

    bundle = st.session_state.get(f"{key_prefix}_bundle")
    if not bundle:
        return

    st.caption(_tt(
        f"Derniere generation : {bundle.get('generated_at_utc','?')} • Entrees AST : {bundle.get('record_count','?')} • Longlist : {bundle.get('longlist_rows','?')} lignes / {bundle.get('longlist_domains','?')} domaines",
        f"Last generation: {bundle.get('generated_at_utc','?')} • AST entries: {bundle.get('record_count','?')} • Longlist: {bundle.get('longlist_rows','?')} rows / {bundle.get('longlist_domains','?')} domains"
    ))

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            _tt("Telecharger FR - version repondant", "Download FR - respondent version"),
            data=bundle["fr_resp"],
            file_name="questionnaire_app22_repondant_FR_v2.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_dl_fr_resp",
        )
        st.download_button(
            _tt("Telecharger EN - version respondent", "Download EN - respondent version"),
            data=bundle["en_resp"],
            file_name="questionnaire_app22_respondent_EN_v2.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_dl_en_resp",
        )
    with d2:
        st.download_button(
            _tt("Telecharger FR - version technique", "Download FR - technical version"),
            data=bundle["fr_tech"],
            file_name="questionnaire_app22_technique_FR_v2.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_dl_fr_tech",
        )
        st.download_button(
            _tt("Telecharger EN - version technique", "Download EN - technical version"),
            data=bundle["en_tech"],
            file_name="questionnaire_app22_technical_EN_v2.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_dl_en_tech",
        )

    st.download_button(
        _tt("Telecharger le ZIP v2 (4 PDF + manifeste)", "Download v2 ZIP (4 PDFs + manifest)"),
        data=bundle["zip"],
        file_name="questionnaire_app22_pdf_bundle_v2.zip",
        mime="application/zip",
        key=f"{key_prefix}_dl_zip",
    )
