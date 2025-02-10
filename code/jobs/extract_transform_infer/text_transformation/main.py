import argparse
import os
import pathlib
import pandas as pd
from tqdm import tqdm

from text_transformation import transform_text as tt
from text_transformation import settings_variables_transform as ttsv


def main_transform(path_in:pathlib.Path, path_out:str):
    df = pd.read_pickle(path_in)
    df['prefix'] = df.apply(lambda x: tt.construct_prefix(x.chapter, x.subchapter), axis=1)
    flaggeds_sentences, df_flag = tt.get_all_flags(df)
    dfs= []
    for fichier in df_flag.fichier.unique():
        print("Analyse fichier ", fichier)
        # On récupère les chunks flaggués.
        sents = df_flag[df_flag.fichier==fichier].md_rows_flag.tolist() + df_flag[df_flag.fichier==fichier].mdheaders.tolist()+ df_flag[df_flag.fichier==fichier].mdtext.tolist()
        sents = [x for x in sents if isinstance(x, str)]
        e, r = tt.chunk_ecart_remarques_simplified(sents) # A ce stade là, on a extrait les écarts et remarques avec le contexte. (ie dans le cas d'un tableau avec toute la ligne)
        

        if len(e) > ttsv.MIN_ECART:
            unduplicated_ecart, ee, pose, doublonse, cose = tt.manage_duplicate(e, verbose=False)
            ecart_data = [(ecart, "E", fichier) for ecart in unduplicated_ecart]
        elif len(e)<= ttsv.MIN_ECART:
            ecart_data = [(ecart, "E", fichier) for ecart in e]
        if len(r) > ttsv.MIN_REMARQUE :
            unduplicated_remarque, rr,posr, doublonsr, cosr  = tt.manage_duplicate(r, verbose=False)
            remarque_data = [(remarque, "R", fichier) for remarque in unduplicated_remarque]
        elif len(r)<= ttsv.MIN_REMARQUE:
            remarque_data = [(remarque, "R", fichier) for remarque in r]

        datas = remarque_data + ecart_data
        df_recap_ecart_remarque = pd.DataFrame(datas, columns=["chunk", "type", "fichier"])
        dfs.append(df_recap_ecart_remarque)
    df = pd.concat(dfs)
    df.to_pickle(path_out)


def main():
    parser = argparse.ArgumentParser(description="")
    # Ajouter l'argument --path
    parser.add_argument('--path_in', type=str, required=True, help="Nom du pickle d'entrée (sortie de extraction)")
    parser.add_argument("--path_out", type=str, required=True, help="Nom du pickle de sortie")
    args = parser.parse_args()
    path_in = pathlib.Path(args.path_in)
    path_out = args.path_out
    main_transform(path_in, path_out)

if __name__ == "__main__":
    main()
   