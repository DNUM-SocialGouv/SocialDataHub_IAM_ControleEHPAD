import argparse
import os
import pathlib
from functools import reduce
import importlib.resources

from typing import Dict, List, Tuple, Any
import pandas as pd
from tqdm import tqdm

from inference_llm import data_models as dm
from inference_llm import standard_utils as su
from inference_llm import settings_variables as sv

def main_classif_w_grade(sentences:List[str], need_excel:bool):
    params = su.get_settings_params()
    model_llm = params["globals"]["model_name"]
    client = su.get_client_openai(params["openai"])

    json_schema = dm.Raisonnement.model_json_schema()
    extra_body={"guided_json": json_schema}

    df_qdp = []
    df_ddr = []
    df_og = []
    df_csps = []
    for sent in tqdm(sentences): # A terme ce sera mieux fait, plus propre.
        messages_qdp  = [
        {"role": "system", "content": sv.SYSTEM_CONTENT_QDP_GRADE},
        {
            "role": "user",
            "content": f"""Voici le rapport de l'inspecteur au format markdown :
        {sent}
        """
        } 
        ]
        output_qdp = su.do_inference(client, messages_qdp, llm_params=params["openai_inference"], model_llm=model_llm , extra_body=extra_body)
        grades_qdp = dm.Grade_Results(answers = output_qdp.choices)
        appartenance_0_1_2, llm_answers =  grades_qdp.get_grade_and_answers()
        df_qdp.append([sent, appartenance_0_1_2, llm_answers])
        
        # DDR
        messages_drr = [
        {"role": "system", "content": sv.SYSTEM_DDR_GRADE},
        {
            "role": "user",
            "content": f"""Voici le rapport de l'inspecteur au format markdown :
        {sent}
        """
        } 
        ]

        output_ddr = su.do_inference(client, messages_drr,  llm_params=params["openai_inference"], model_llm=model_llm , extra_body=extra_body)
        grades_ddr = dm.Grade_Results(answers = output_ddr.choices)
        appartenance_0_1_2, llm_answers =  grades_ddr.get_grade_and_answers()
        df_ddr.append([sent, appartenance_0_1_2, llm_answers])

        # CSPS
        messages_csps= [
        {"role": "system", "content": sv.SYSTEM_CSPS_GRADE},
        {
            "role": "user",
            "content": f"""Voici le rapport de l'inspecteur au format markdown :
        {sent}
        """
        } 
        ]
        output_csps = su.do_inference(client, messages_csps, llm_params=params["openai_inference"], model_llm=model_llm , extra_body=extra_body)
        print(output_csps.choices)
        grades_csps = dm.Grade_Results(answers=output_csps.choices)
        appartenance_0_1_2, llm_answers =  grades_csps.get_grade_and_answers()
        df_csps.append([sent, appartenance_0_1_2, llm_answers])

        #OG
        messages_og = [{"role": "system", "content": sv.SYSTEM_OG_GRADE},
        {
            "role": "user",
            "content": f"""Voici le rapport de l'inspecteur au format markdown :
        {sent}
        """
        } 
        ]
        output_og = su.do_inference(client, messages_og,  llm_params=params["openai_inference"], model_llm=model_llm , extra_body=extra_body)
        grades_og = dm.Grade_Results(answers=output_og.choices)
        appartenance_0_1_2, llm_answers =  grades_og.get_grade_and_answers()
        df_og.append([sent, appartenance_0_1_2, llm_answers])


    df_qdp = pd.DataFrame(df_qdp, columns=["chunks", "appartenance_0_1_2_QDP", "llm_answers_QDP"])
    df_ddr = pd.DataFrame(df_ddr, columns=["chunks", "appartenance_0_1_2_DDR", "llm_answers_DDR"])
    df_csps = pd.DataFrame(df_csps, columns=["chunks", "appartenance_0_1_2_CSPS", "llm_answers_CSPS"])
    df_og = pd.DataFrame(df_og, columns=["chunks", "appartenance_0_1_2_OG", "llm_answers_OG"])

    df_merge = reduce(lambda left,right : pd.merge(left, right, on="chunks", how="outer"), [df_qdp, df_ddr, df_csps, df_og])
    if need_excel:
        df_merge.to_excel("df_subset_grade.xlsx") # Permet d'écrire le résultat sous forme d'Excel.

    

def main():
    parser = argparse.ArgumentParser(description="")
    # Ajouter l'argument --path
    parser.add_argument('--extract_excel', type=bool, required=True, help="Extraction du résultat sous forme d'Excel.")
    args = parser.parse_args()

    sentences = ["Test phrase1 .", " En l’absence de procédure d’admission et d’accueil des nouveaux salariés/stagiaires/bénévoles, l’intégration de ces derniers et leur adaptation à l’emploi n’est pas facilitée ce qui est susceptible d’affecter la qualité de l’accompagnement des résidents."]
    
    main_classif_w_grade(sentences, need_excel=args.extract_excel)


if __name__ == "__main__":
    main()
   