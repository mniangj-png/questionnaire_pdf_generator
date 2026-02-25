# -*- coding: utf-8 -*-
"""
App séparée Streamlit pour générer les PDF du questionnaire app22 (FR/EN, Répondant/Technique)
sans modifier l'application de collecte en cours.

Dépendances principales :
- streamlit
- pandas
- reportlab
- openpyxl (si lecture de longlist.xlsx)
- questionnaire_pdf_export_v2.py (module fourni séparément, à placer dans le même dossier)

Usage local :
    streamlit run app_questionnaire_pdf_generator_separate.py

Déploiement Streamlit Community Cloud (GitHub) :
- placer ce fichier + questionnaire_pdf_export_v2.py dans le repo
- ajouter éventuellement l'app22 et la longlist au repo (ou les téléverser depuis l'interface)
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

try:
    from questionnaire_pdf_export_v2 import build_all_questionnaire_pdfs
except Exception as e:  # pragma: no cover
    build_all_questionnaire_pdfs = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


DEFAULT_APP22_PATH = "/mnt/data/app22_offline_local_sqlite_v3.py"
DEFAULT_LONGLIST_PATH = "/mnt/data/longlist.xlsx"


@dataclass
class PreparedInputs:
    workdir: str
    app_path: str
    longlist_path: Optional[str]
    cleanup: bool = False


def _init_state():
    st.session_state.setdefault("pdfgen_bundle", None)
    st.session_state.setdefault("pdfgen_manifest", None)
    st.session_state.setdefault("pdfgen_last_inputs", None)


def _check_module_ready():
    if build_all_questionnaire_pdfs is None:
        st.error(
            "Le module `questionnaire_pdf_export_v2.py` est introuvable ou ne peut pas être importé. "
            "Placez-le dans le même dossier que cette app."
        )
        if _IMPORT_ERROR is not None:
            st.exception(_IMPORT_ERROR)
        st.stop()


def _save_uploaded_file(uploaded_file, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def _prepare_from_uploads(app22_upload, longlist_upload) -> PreparedInputs:
    tmpdir = tempfile.mkdtemp(prefix="app22_pdfgen_")
    workdir = Path(tmpdir)

    app_name = app22_upload.name if app22_upload else "app22_uploaded.py"
    if not app_name.lower().endswith(".py"):
        app_name = f"{app_name}.py"
    app_path = workdir / app_name
    _save_uploaded_file(app22_upload, app_path)

    longlist_path = None
    if longlist_upload is not None:
        lname = longlist_upload.name
        # Le module v2 cherche automatiquement longlist.xlsx / indicator_longlist.csv
        suffix = Path(lname).suffix.lower()
        if suffix == ".csv":
            target_name = "indicator_longlist.csv"
        else:
            target_name = "longlist.xlsx"
        ll_path = workdir / target_name
        _save_uploaded_file(longlist_upload, ll_path)
        longlist_path = str(ll_path)

    return PreparedInputs(workdir=str(workdir), app_path=str(app_path), longlist_path=longlist_path, cleanup=True)


def _prepare_from_repo_paths(app_path_str: str, longlist_path_str: Optional[str]) -> PreparedInputs:
    app_path = Path(app_path_str).expanduser().resolve()
    if not app_path.exists():
        raise FileNotFoundError(f"Fichier app22 introuvable : {app_path}")

    tmpdir = tempfile.mkdtemp(prefix="app22_pdfgen_")
    workdir = Path(tmpdir)

    staged_app = workdir / app_path.name
    shutil.copy2(app_path, staged_app)

    staged_longlist = None
    if longlist_path_str:
        ll = Path(longlist_path_str).expanduser().resolve()
        if not ll.exists():
            raise FileNotFoundError(f"Longlist introuvable : {ll}")
        suffix = ll.suffix.lower()
        target_name = "indicator_longlist.csv" if suffix == ".csv" else "longlist.xlsx"
        staged_longlist = workdir / target_name
        shutil.copy2(ll, staged_longlist)

    return PreparedInputs(workdir=str(workdir), app_path=str(staged_app), longlist_path=str(staged_longlist) if staged_longlist else None, cleanup=True)


def _cleanup_prepared(prepared: Optional[PreparedInputs]):
    if not prepared or not prepared.cleanup:
        return
    try:
        shutil.rmtree(prepared.workdir, ignore_errors=True)
    except Exception:
        pass


def _render_manifest_summary(manifest: dict):
    st.subheader("Résumé de la génération")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rubriques détectées", manifest.get("sections_count"))
    with c2:
        st.metric("Libellés détectés", manifest.get("labels_count"))
    with c3:
        st.metric("Fichiers générés", len(manifest.get("files", [])))

    ll = manifest.get("longlist", {}) or {}
    with st.expander("Détails longlist (annexe)", expanded=True):
        st.write({
            "trouvée": ll.get("found"),
            "source_path": ll.get("source_path"),
            "nb_lignes": ll.get("row_count"),
            "nb_domaines": ll.get("domain_count"),
            "erreur": ll.get("error"),
        })

    with st.expander("Fichiers générés", expanded=False):
        for f in manifest.get("files", []):
            st.write(f"• {f}")


def _bundle_to_download_buttons(bundle: dict):
    st.subheader("Téléchargement des livrables")

    # PDFs
    name_map = [
        ("questionnaire_app22_repondant_FR.pdf", "Télécharger FR - version répondant"),
        ("questionnaire_app22_technique_FR.pdf", "Télécharger FR - version technique"),
        ("questionnaire_app22_respondent_EN.pdf", "Download EN - respondent version"),
        ("questionnaire_app22_technical_EN.pdf", "Download EN - technical version"),
        ("manifest_questionnaire_pdf_export_v2.json", "Télécharger le manifeste JSON"),
    ]

    for key, label in name_map:
        if key in bundle:
            mime = "application/pdf" if key.endswith(".pdf") else "application/json"
            st.download_button(
                label=label,
                data=bundle[key],
                file_name=key,
                mime=mime,
                use_container_width=True,
                key=f"dl_{key}",
            )

    # ZIP
    if "zip" in bundle:
        st.download_button(
            label="Télécharger le ZIP (4 PDF + manifeste)",
            data=bundle["zip"],
            file_name="questionnaire_app22_pdf_bundle_v2.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_zip_bundle_v2",
        )


def _build_zip_from_bundle(bundle_files: dict) -> bytes:
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in bundle_files.items():
            z.writestr(name, data)
    buf.seek(0)
    return buf.getvalue()


def _generate_bundle(prepared: PreparedInputs) -> Tuple[dict, dict]:
    files = build_all_questionnaire_pdfs(prepared.app_path)
    # Le module renvoie déjà le manifeste JSON dans les fichiers
    manifest = {}
    if "manifest_questionnaire_pdf_export_v2.json" in files:
        try:
            manifest = json.loads(files["manifest_questionnaire_pdf_export_v2.json"].decode("utf-8"))
        except Exception:
            manifest = {}
    bundle = dict(files)
    bundle["zip"] = _build_zip_from_bundle(files)
    return bundle, manifest


def main():
    st.set_page_config(page_title="Générateur PDF app22 (séparé)", page_icon="📄", layout="wide")
    _init_state()
    _check_module_ready()

    st.title("Générateur séparé des PDF du questionnaire app22")
    st.caption(
        "Cette app est indépendante de l’app22 de collecte : elle n’écrit rien dans la base et ne modifie pas le code de collecte."
    )

    st.info(
        "Entrées attendues : (1) le fichier Python de l’app22, (2) la longlist (Excel .xlsx ou CSV). "
        "Sorties : 4 PDF (FR/EN × Répondant/Technique) + un manifeste JSON + un ZIP."
    )

    mode = st.radio(
        "Mode d’alimentation",
        ["Téléverser les fichiers (recommandé)", "Utiliser des chemins de fichiers déjà présents sur le serveur"],
        horizontal=True,
    )

    prepared = None
    generated = False

    with st.form("pdfgen_form"):
        if mode == "Téléverser les fichiers (recommandé)":
            app22_upload = st.file_uploader(
                "Fichier app22 (.py)",
                type=["py"],
                help="Ex. : app22_offline_local_sqlite_v3.py",
                key="u_app22",
            )
            longlist_upload = st.file_uploader(
                "Longlist (.xlsx ou .csv)",
                type=["xlsx", "csv"],
                help="Ex. : longlist.xlsx ou indicator_longlist.csv",
                key="u_longlist",
            )

            submitted = st.form_submit_button("Générer les PDF")
            if submitted:
                if app22_upload is None:
                    st.error("Veuillez téléverser le fichier Python de l’app22.")
                elif longlist_upload is None:
                    st.error("Veuillez téléverser la longlist (.xlsx ou .csv).")
                else:
                    try:
                        prepared = _prepare_from_uploads(app22_upload, longlist_upload)
                        bundle, manifest = _generate_bundle(prepared)
                        st.session_state["pdfgen_bundle"] = bundle
                        st.session_state["pdfgen_manifest"] = manifest
                        st.session_state["pdfgen_last_inputs"] = {
                            "mode": "uploads",
                            "app_file": app22_upload.name,
                            "longlist_file": longlist_upload.name,
                        }
                        generated = True
                    finally:
                        _cleanup_prepared(prepared)

        else:
            app_path_str = st.text_input(
                "Chemin du fichier app22 (.py)",
                value=DEFAULT_APP22_PATH if Path(DEFAULT_APP22_PATH).exists() else "",
                placeholder="/path/to/app22_offline_local_sqlite_v3.py",
            )
            longlist_path_str = st.text_input(
                "Chemin de la longlist (.xlsx ou .csv)",
                value=DEFAULT_LONGLIST_PATH if Path(DEFAULT_LONGLIST_PATH).exists() else "",
                placeholder="/path/to/longlist.xlsx",
            )

            submitted = st.form_submit_button("Générer les PDF")
            if submitted:
                try:
                    prepared = _prepare_from_repo_paths(app_path_str, longlist_path_str or None)
                    bundle, manifest = _generate_bundle(prepared)
                    st.session_state["pdfgen_bundle"] = bundle
                    st.session_state["pdfgen_manifest"] = manifest
                    st.session_state["pdfgen_last_inputs"] = {
                        "mode": "paths",
                        "app_path": app_path_str,
                        "longlist_path": longlist_path_str,
                    }
                    generated = True
                except Exception as e:
                    st.error(f"Erreur : {e}")
                finally:
                    _cleanup_prepared(prepared)

    if generated:
        st.success("Génération terminée avec succès.")

    bundle = st.session_state.get("pdfgen_bundle")
    manifest = st.session_state.get("pdfgen_manifest")
    last_inputs = st.session_state.get("pdfgen_last_inputs")

    if last_inputs:
        with st.expander("Dernières entrées utilisées", expanded=False):
            st.json(last_inputs)

    if manifest:
        _render_manifest_summary(manifest)

    if bundle:
        _bundle_to_download_buttons(bundle)

    with st.expander("Conseils de déploiement (GitHub + Streamlit)", expanded=False):
        st.markdown(
            """
- Déployez **cette app séparée** dans un dépôt GitHub distinct (ou un sous-dossier distinct).
- Placez `questionnaire_pdf_export_v2.py` dans le même dossier.
- Dépendances minimales dans `requirements.txt` :
  - `streamlit`
  - `pandas`
  - `reportlab`
  - `openpyxl`
- Vous pouvez **téléverser** `app22_offline_local_sqlite_v3.py` et `longlist.xlsx` directement depuis l’interface, sans toucher à l’app22 en production.
            """
        )


if __name__ == "__main__":
    main()
