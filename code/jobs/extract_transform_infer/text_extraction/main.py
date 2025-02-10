import argparse
import os
import pathlib
from text_extraction import preprocess_text as pt
import nltk
nltk.download('stopwords')

def main_extract(base_path: pathlib.Path, output_excel: str):
    dfs = []
    for file in os.listdir(base_path):
        if not file.endswith('.docx'):  # Filtrer uniquement les fichiers .docx
            print(f"Fichier ignoré : {file}")
            continue

        print(f"{file} en cours de traitement.")
        current_file_path = pathlib.Path(base_path, file)

        try:
            results = pt.extract_word_info(current_file_path)
            assert len(results["content_raw"]) == len(results["headings"]), "content_raw et headings sont de taille différentes"
            df = pt.liste_vers_dataframe(results["content_raw"])
            df["chapter"] = [x[0] for x in results["headings"]]
            df["subchapter"] = [x[1] for x in results["headings"]]
            df["fichier"] = [file] * len(df)
            dfs.append(df)
            print(f"{file} traité")
        except Exception as e:
            print(f"Erreur lors du traitement de {file} : {e}")

    if dfs:  # Vérifiez qu'il y a des données avant d'écrire dans le fichier
        pt.concat_listes_df_and_save_on_pickle(dfs, output_excel)
        print("Excel créé")
    else:
        print("Aucun fichier valide n'a été traité.")

def main():
    parser = argparse.ArgumentParser(description="")
    # Ajouter l'argument --path
    parser.add_argument('--path_folder', type=str, required=True, help='Chemin du dossier contenant les fichiers à extraire.')
    parser.add_argument("--output_excel", type=str, required=True, help="Nom du fichier excel représentant l'extraction")
    args = parser.parse_args()
    base_path = pathlib.Path(args.path_folder)
    output_excel = args.output_excel
    main_extract(base_path, output_excel)

if __name__ == "__main__":
    main()
   