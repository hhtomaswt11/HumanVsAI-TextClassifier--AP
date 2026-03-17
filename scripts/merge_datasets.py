import pandas as pd
import os
import csv


def _read_dataset(path):
    # Prefer ';' as requested, but keep a ',' fallback for mixed legacy files.
    for sep in (';', ','):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
            if not df.empty and len(df.columns) >= 2:
                return df
        except Exception:
            continue
    raise ValueError(f'Could not parse CSV file: {path}')


def _normalize_columns(df):
    # Normalize common variants from older datasets.
    rename_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if low == 'text':
            rename_map[col] = 'text'
        elif low == 'label':
            rename_map[col] = 'label'
    return df.rename(columns=rename_map)


def _clean_text_column(df):
    if 'text' not in df.columns:
        return df

    # Remove outer double-quote delimiters if they are present in the value.
    df['text'] = (
        df['text']
        .astype(str)
        .str.strip()
        .str.replace(r'^"(.*)"$', r'\1', regex=True)
        .str.replace('""', '"', regex=False)
    )
    return df


def merge_datasets():
    HUMAN_FILE = 'dataset_human.csv'
    GOOGLE_FILE = 'dataset_google.csv'
    META_FILE = 'dataset_meta.csv'
    OPENAI_TEXT_FILE = 'dataset_openai.csv'
    ANTHROPIC_TEXT_FILE = 'dataset_anthropic.csv'
    OUTPUT_FILE = 'dataset_limpo.csv'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')

    # Prefer datasets in /data, but keep script_dir as fallback.
    search_dirs = [data_dir, script_dir]

    files = [HUMAN_FILE, GOOGLE_FILE, META_FILE, OPENAI_TEXT_FILE, ANTHROPIC_TEXT_FILE]

    dfs = []
    for fname in files:
        path = next((os.path.join(d, fname) for d in search_dirs if os.path.exists(os.path.join(d, fname))), None)
        if not path:
            print(f'File not found: {fname} (searched in {search_dirs})')
            continue

        _, ext = os.path.splitext(fname.lower())
        try:
            if ext == '.csv':
                df = _read_dataset(path)
                df = _normalize_columns(df)
                df = _clean_text_column(df)
                dfs.append(df)
        except Exception:
            # if reading fails, try a simple text read
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines()]
                lines = [l for l in lines if l]
                if lines:
                    df = pd.DataFrame({'text': lines})
                    dfs.append(df)
            except Exception:
                # give up on this file
                continue

    if not dfs:
        print('No dataset files found to merge.')
        return

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = os.path.join(data_dir, OUTPUT_FILE)
    # Save CSV with ';' delimiter as requested.
    combined.to_csv(out_path, sep=';', index=False, encoding='utf-8')
    print(f'Merged {len(dfs)} files into {out_path} with {len(combined)} rows.')


if __name__ == '__main__':
    merge_datasets()


