from pathlib import Path
import hashlib
import json
import zipfile


def make_run_zip(run_dir,include_predictions=True):
    run_dir=Path(run_dir).resolve()
    result=run_dir.parent/(run_dir.name+'.zip')
    with zipfile.ZipFile(result,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(run_dir.rglob('*')):
            if not p.is_file():continue
            if not include_predictions and 'predictions' in p.parts:continue
            z.write(p,p.relative_to(run_dir))
    print('Archive Kaggle :',result,flush=True)
    return result


def safe_extract(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path=(destination/info.filename).resolve()
            if not path.is_relative_to(destination):
                raise ValueError('Chemin malveillant détecté dans le ZIP')
            # Rejeter les liens symboliques Unix
            if ((info.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('Lien symbolique non autorisé dans le ZIP')
        z.extractall(destination)
