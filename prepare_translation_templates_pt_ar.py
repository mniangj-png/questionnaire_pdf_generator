from pathlib import Path
import pandas as pd

BASE = Path('.')
DATA_DIR = BASE / 'data'

LONG_CSV_CANDIDATES = [
    DATA_DIR / 'indicator_longlist.csv',
    BASE / 'indicator_longlist.csv',
]
LONG_XLSX_CANDIDATES = [
    DATA_DIR / 'longlist.xlsx',
    BASE / 'longlist.xlsx',
]
COUNTRY_XLSX_CANDIDATES = [
    DATA_DIR / 'COUNTRY_ISO3_with_EN.xlsx',
    BASE / 'COUNTRY_ISO3_with_EN.xlsx',
]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def prep_longlist_csv(path: Path):
    df = pd.read_csv(path, dtype=str).fillna('')
    # Expected CSV columns in app22: domain_code, domain_label_fr, domain_label_en, stat_code, stat_label_fr, stat_label_en
    if 'domain_label_pt' not in df.columns:
        df['domain_label_pt'] = df.get('domain_label_en', df.get('domain_label_fr', ''))
    if 'domain_label_ar' not in df.columns:
        df['domain_label_ar'] = df.get('domain_label_en', df.get('domain_label_fr', ''))
    if 'stat_label_pt' not in df.columns:
        df['stat_label_pt'] = df.get('stat_label_en', df.get('stat_label_fr', ''))
    if 'stat_label_ar' not in df.columns:
        df['stat_label_ar'] = df.get('stat_label_en', df.get('stat_label_fr', ''))

    out = path.with_name(path.stem + '_pt_ar_template.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    return out


def prep_longlist_xlsx(path: Path):
    df = pd.read_excel(path, dtype=str).fillna('')
    # Expected XLSX columns in app22: Domain_code, Domain_label_fr, Domain_label_en, Stat_label_fr, Stat_label_en
    if 'Domain_label_pt' not in df.columns:
        df['Domain_label_pt'] = df.get('Domain_label_en', df.get('Domain_label_fr', ''))
    if 'Domain_label_ar' not in df.columns:
        df['Domain_label_ar'] = df.get('Domain_label_en', df.get('Domain_label_fr', ''))
    if 'Stat_label_pt' not in df.columns:
        df['Stat_label_pt'] = df.get('Stat_label_en', df.get('Stat_label_fr', ''))
    if 'Stat_label_ar' not in df.columns:
        df['Stat_label_ar'] = df.get('Stat_label_en', df.get('Stat_label_fr', ''))

    out = path.with_name(path.stem + '_pt_ar_template.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='longlist')
    return out


def prep_country_xlsx(path: Path):
    df = pd.read_excel(path, dtype=str).fillna('')
    if 'COUNTRY_NAME_PT' not in df.columns:
        df['COUNTRY_NAME_PT'] = df.get('COUNTRY_NAME_EN', df.get('COUNTRY_NAME_FR', ''))
    if 'COUNTRY_NAME_AR' not in df.columns:
        df['COUNTRY_NAME_AR'] = df.get('COUNTRY_NAME_EN', df.get('COUNTRY_NAME_FR', ''))

    out = path.with_name(path.stem + '_PT_AR_template.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='countries')
    return out


def main():
    created = []

    long_csv = first_existing(LONG_CSV_CANDIDATES)
    if long_csv:
        created.append(prep_longlist_csv(long_csv))

    long_xlsx = first_existing(LONG_XLSX_CANDIDATES)
    if long_xlsx:
        created.append(prep_longlist_xlsx(long_xlsx))

    country_xlsx = first_existing(COUNTRY_XLSX_CANDIDATES)
    if country_xlsx:
        created.append(prep_country_xlsx(country_xlsx))

    if not created:
        print('Aucun fichier source trouvé. Placez les fichiers dans le dépôt, puis relancez ce script.')
        return

    print('Fichiers générés :')
    for p in created:
        print('-', p)

    print('\nÉtape suivante : traduire les nouvelles colonnes PT/AR, puis remplacer les fichiers d’origine dans data/.')


if __name__ == '__main__':
    main()
