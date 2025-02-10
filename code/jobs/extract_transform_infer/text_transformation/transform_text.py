from os import name
from typing import Deque, List, Optional, Tuple, Any, Dict, Annotated
import json
import importlib.resources
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np
import torch

import pandas as pd

from text_extraction import preprocess_text as tept
from text_extraction import settings_variables as tesv
from text_transformation import settings_variables_transform as ttsv

def from_list_to_markdown(liste:list[str])->List[str]:
    sentences = []
    for sents in liste :
        if isinstance(sents, str): # Dans le cas du texte brut
            md_sent = "|" + " | ".join(liste) + "|"
            sentences+= [md_sent]
        elif isinstance(sents, list):
            md_sent = "|" + " | ".join(sents) + "|"
            sentences+= [md_sent]
    sentences = sentences
    return sentences

def construct_prefix(chap:str, subchap:str) -> str:
    """Cette fonction permet de construire le prefix markdown chapitre et sous chapitre."""
    if subchap == "" and len(chap)>0 : # Uniquement chapitre
        return f"#{chap}\n"
    if chap == "" and len(subchap)>0: # Uniquement un sous chapitre
        return f"#{subchap}\n"
    if len(chap)>0 and len(subchap)>0:
        return f"#{chap}\n##{subchap}\n"
    return ""

def deduplicate_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def get_all_flags(df:pd.DataFrame, col_plain_text = "text", col_flag="flag", col_header = "headers", col_prefix="prefix", prefix_md="md",col_origine="origine") -> Tuple[List[str], pd.DataFrame]:
    """Cette fonction permet de réprer le contenu qui nous intéresse (écart ou remarque dans le cas de DataVal)
    puis de le transformer en markdown.
    
    Arguments :
    df : DataFrame qui représente les informations extraites block par block en sortie de extract_word_info (de la brique text_extraction en gros)
    prefix_md : prefix à rajouter devant les chunks transformés en markdown.
    les autres arguments : des noms de colonnes.

    Returns :  
    prompt_sentences : liste des chunks 
    df_whole : Dataframe des chunks flaggués

    """
    def extraire_elements(row):
        indices = row['true_flag_position']
        if len(indices) > 0:
            valeurs = row['data']
            flag_lines =  [valeurs[i] for i in indices]
            flag_lines_md = from_list_to_markdown(flag_lines)
            header_md = from_list_to_markdown(row[col_header])[0] # Créer la ligne markdown
            ecart_in_markdown = tept.contient_motif(header_md)
            if ecart_in_markdown :# Si le markdown contient un regex écart, en fait c'est un markdown Word mais pas sémantique. On le retire.
                header_md = header_md + "\n" + "----|"*len(row[col_header]) +"\n" # contenu de l'entête + séparation markdown
                prefix_md = row[col_prefix]+"\n"*bool(len(row[col_prefix])) # prefix + saut de ligne
                
                return [f"{prefix_md}{line}" for line in flag_lines_md]
            else:
                header_md = header_md + "\n" + "----|"*len(row[col_header]) +"\n" # contenu de l'entête + séparation markdown
                prefix_md = row[col_prefix]+"\n"*bool(len(row[col_prefix])) # prefix + saut de ligne
                
                return [f"{prefix_md}{header_md}{line}" for line in flag_lines_md]
        else:
            return None

    col_md_text: str = prefix_md + col_plain_text
    col_md_header: str = prefix_md + col_header
    col_md_rows_table: str = prefix_md + "_rows_" + col_flag
    
    line_caracter = "\n"


    # Récupère les flags de "texte plein"
    prompt_sentences:list[str] = [] # Cette liste répresente le contenu correspondant au texte à envoyer dans le prompt
    df_text_flag = df[df[col_flag]==True].copy()
    df_text_flag[col_md_text] = df[col_prefix] + df[col_plain_text] 

    # Récupère les flags de headers.
    df_table = df[df[col_header].apply(lambda x:isinstance(x,list))]
    df_table_flag_header  = df_table[df_table[col_flag].apply(lambda x:x[0])] # DataFrame qui contient uniquement les headers flagués True
    print("len table", len(df_table[col_prefix].tolist()))
    sent_headers = [f"{x}{line_caracter*bool(len(x))}{y}" for (x,y) in zip(df_table_flag_header[col_prefix].tolist(), from_list_to_markdown(df_table_flag_header[col_header].tolist()))]
    print("len new", len(sent_headers))
    df_table_flag_header[col_md_header] = sent_headers
    # Récupère les flags de lignes de tableaux.
    df_table["true_flag_position"] = df_table[col_flag].apply(lambda t:[i for i, x in enumerate(t[1:]) if x] if isinstance(t, list) else None)

    

    rows_flag = df_table.apply(extraire_elements,axis=1)
    df_table[col_md_rows_table] = rows_flag

    df_table.dropna(subset=col_md_rows_table, inplace=True)
    
    exploded_df_table = df_table.explode(column=col_md_rows_table)

    # Ajoute l'origine au fichier.
    def assign_column_or_notify(df, column, value, empty_message):
        if not df.empty:
            df.loc[:, column] = value
        else:
            print(empty_message)
    dataframes_params = [
    (df_text_flag, col_origine, col_plain_text, "Pas de text extrait, assez rare"),
    (df_table_flag_header, col_origine, col_header, "Pas de header extrait"),
    (exploded_df_table, col_origine, "table", "Pas de tableau extrait, assez rare")
    ]
    for df, column, value, message in dataframes_params:
        assign_column_or_notify(df, column, value, message)


    whole_df = pd.concat([df_text_flag, df_table_flag_header, exploded_df_table])


    prompt_sentences = df_text_flag[col_md_text].tolist() + sent_headers + rows_flag.explode().tolist()

    return prompt_sentences, whole_df

def split_text(text):
    """
    Découpe le texte à chaque occurrence des mots-clés "ecart", "e", "remarque", ou "r".
    Chaque segment résultant commence par l'un de ces mots-clés.

    Args:
        text (str): Le texte à découper.

    Returns:
        List[str]: Une liste de segments découpés.
    """
    regex_entete = tesv.REGEX_ENTETE
    regex_split = r'(\b|\s)(ecart\d{0,2}|écart\d{0,2}|e\d{0,2}|remarque\d{0,2}|r\s*\d{1,2}|r\s+\:+)(\b|\s)'
    
    pattern = re.compile(regex_entete + regex_split, re.IGNORECASE )
    
    # Utiliser re.split avec des groupes capturants pour conserver les délimiteurs
    parts = pattern.split(text)
    
    result = []
    
    if parts:
        # Le premier élément est le texte avant tout mot-clé
        before = parts[0].strip()
        if before:
            result.append(before)
        
        # Itérer sur les paires (mot-clé, texte après le mot-clé)
        for i in range(1, len(parts), 2):
            keyword = parts[i].strip()
            # Vérifier s'il y a du texte après le mot-clé
            after = parts[i+1].strip() if i+1 < len(parts) else ''
            # Combiner le mot-clé avec le texte suivant
            combined = f"{keyword} {after}" if after else keyword
            result.append(combined)
    
    return result[1:]

def deal_overlap(spans_entitiesA:List[List[int]], spans_entitiesB:List[List[int]]):
    """
    Cette fonction permet de gérer les cas où il y aurait des overlaps.
    Dans notre cas entités A sont les remarques, entités B les écarts.
    On va itérer sur toutes les paires. Le plus simple, c'est de faire une double boucle.
    Cependant, comme chaque ajustement peut impacter plusieurs spans, il peut être nécessaire
    de répéter l'opération jusqu'à ce qu'il n'y ait plus d'overlap.
    
    Pour éviter une boucle infinie, on fait un maximum de quelques itérations,
    ou on vérifie après un passage si plus aucun overlap n’est présent.


    """
    # On s'assure que les écarts et les remarques sont triés par ordre de début
    spans_entitiesA = sorted(spans_entitiesA, key=lambda x: x[0])
    spans_entitiesB = sorted(spans_entitiesB, key=lambda x: x[0])


    changed = True
    while changed:
        changed = False
        new_ecarts = []
        for (e_start, e_end) in spans_entitiesA:
            # On va vérifier cet entitéA avec toutes les entitésB
            e_span_modified = (e_start, e_end)
            for i, (r_start, r_end) in enumerate(spans_entitiesB):
                # Vérif de l'overlap
                if e_span_modified[0] <= r_end and r_start <= e_span_modified[1]: # Overlap détecté
                    e_s, e_e = e_span_modified
                    if e_s >= r_start and e_s <= r_end: # L'écart commence pendant la remarque
                        # On coupe la remarque
                        new_r_end = e_s - 1
                        if new_r_end < r_start:
                            # La remarque est invalide, on la supprime
                            spans_entitiesB[i] = None
                        else:
                            spans_entitiesB[i] = (r_start, new_r_end)
                        changed = True
                    else:
                        # L'écart commence avant la remarque
                        # On coupe l'écart
                        new_e_end = r_start - 1
                        if new_e_end < e_s:
                            # L'écart devient invalide
                            e_span_modified = None
                            break
                        else:
                            e_span_modified = (e_s, new_e_end)
                        changed = True
            
            # Après avoir traité toutes les remarques, on garde l’écart s’il est toujours valide
            if e_span_modified is not None:
                new_ecarts.append(e_span_modified)

        # On nettoie les listes : on enlève les None
        entitiesA = new_ecarts
        
        spans_entitiesB = [tuple(x) for x in spans_entitiesB] # Conversion en tuple une itération réussieu
        entitiesB = [r for r in spans_entitiesB if r is not None]

        return entitiesA, entitiesB





def chunk_ecart_remarques_simplified(candidats:List[str], verbose=ttsv.VERBOSE) -> Tuple[List, List]:
    """
    A partir d'une lsite de chunk cette fonction les chunks écarts et remarques.
    """
    regex_entete = tesv.REGEX_ENTETE
    regex_ecart = regex_entete + r"\b(\becart\b|\bécart\b|\be\b).+(\n-.+)*"
    regex_remarque = regex_entete + r"\b(\bremarque\b|\bremarques\b|\br(?!(\.|\s+\d+-\d+))\b).+(\n-.+)*" # Rnegativ lookhead suivant R car  y'a les codes CSF R.2354/ R 256-89
    ecarts:List[str] = []
    remarques:List[str] = []
    for candidat in candidats:
        spans_ecarts:List = []
        spans_remarques:List = []
        finditer_ecart = re.finditer(regex_ecart, candidat,  flags=re.IGNORECASE)
        finditer_remarque = re.finditer(regex_remarque, candidat,  flags=re.IGNORECASE)
        for find in finditer_ecart:
            spans_ecarts.append(list(find.span()))
        for find in finditer_remarque:
            spans_remarques.append(list(find.span()))

        # Regardons si jamais ils s'overlapent et corrigeons ça
        spans_ecarts, spans_remarques= deal_overlap(spans_ecarts, spans_remarques )
        fixing_caracters = "µ"
        for i_ecart, span_ecart in enumerate(spans_ecarts):
            current_candidat = candidat
            other_ecarts_remarques = spans_ecarts[:i_ecart] + spans_ecarts[i_ecart+1:] + spans_remarques # Tout ce qui n'est pas l'ecart en court
            for other_ in other_ecarts_remarques:
                debut = other_[0]
                fin = other_[1]
                partie_avant = current_candidat[:debut]
                # Partie à remplacer par 'µ'
                partie_remplacee = fixing_caracters * (fin - debut)
                # Partie après la zone remplacée
                partie_apres = current_candidat[fin:]
                current_candidat = partie_avant + partie_remplacee + partie_apres
            current_candidat = current_candidat.replace(fixing_caracters, "")
            ecarts.append(current_candidat)
        for i_remarque, span_remarque in enumerate(spans_remarques):
            other_remarques_ecarts = list(spans_remarques[:i_remarque]) + list(spans_remarques[i_remarque+1:]) + spans_ecarts # Miroir
            current_candidat = candidat
            for _ in other_remarques_ecarts:
                debut = _[0]
                fin = _[1]
                partie_avant = current_candidat[:debut]
                # Partie à remplacer par 'µ'
                partie_remplacee = fixing_caracters * (fin - debut)
                # Partie après la zone remplacée
                partie_apres = current_candidat[fin:]
                current_candidat = partie_avant + partie_remplacee + partie_apres
            current_candidat = current_candidat.replace(fixing_caracters, "")
            remarques.append(current_candidat)

    return ecarts, remarques

def spot_doublons(sentences:List[str], vectorizer, verbose=ttsv.VERBOSE, threshold_overlap_p=ttsv.THRESHOLD_OVERLAP_P) -> Tuple[List[int], Dict[str, str], Any]:
    """
    Cette fonction renvoie un dictionnaire des doublons selon la cosine TFIDF.

    Arguments
    -----------------
    sentences : list des chunks à analyser.
    threshold_overlap_p : threshold à partir duquel un overlap est considéré valide.

    Returns 
    ---------------
    liste des positions des doublons
    Le dictionnaire retournée est Dict[keep_sentence] : throwed_sentences
    """
    X_ = vectorizer.fit_transform(sentences)
    o = cosine_similarity(X_, X_)
    tri = np.triu(o, k=1)
    mask = tri == 0
    non_zero_triangulaire_element = tri[~mask] # On récupère tout ce qui n'est pas 0
    std = np.std(non_zero_triangulaire_element)
    nu = np.mean(non_zero_triangulaire_element)
    doublons = dict()
    positions_doublons = []
    for i_anchor in range(len(sentences)):
        top3 = torch.topk(torch.tensor(o[i_anchor, :]), ttsv.MINIMAL) # 
        indices = top3.indices
        values = top3.values
        mask_no_anchor = indices!=i_anchor # En cas d'exact doublon il peut y avoir deux cosine à 1, la phrase elle même et son doublon donc.
        top_candidat = (values[mask_no_anchor][0], indices[mask_no_anchor][0]) # (cosine, indice du doublon). 
        sentence_candidat_doublon = sentences[top_candidat[1]]
        sentence_candidat = sentences[i_anchor]
        if top_candidat[0]>=nu+3*std: # cosine anormalement proche

            if sentence_candidat in list(doublons.values()) : # La phrase analysée est déjà considérée comme un doublon.
                if verbose:
                    print("Already un doublon ", sentence_candidat_doublon[0:100])
                    print("-"*10)
                
            else:
                # Est-ce que le %age d'overlap entre les deux est assez grand ?
                overlap_p = sum([x.size/min(len(sentence_candidat), len(sentence_candidat_doublon)) \
                    for x in SequenceMatcher(None, sentence_candidat, sentence_candidat_doublon).get_matching_blocks()])
                if overlap_p >threshold_overlap_p:
                    doublons[sentence_candidat] = sentence_candidat_doublon
                    positions_doublons.append(int(top_candidat[1]))
                    if verbose:
                        print("anchor :\n", sentence_candidat[0:100])
                        print("a pour doublon OVERLAP :", sentence_candidat_doublon[0:100])
                        print("Valeur de la cosine ", top_candidat[0], " et la valeur du threshold ", nu + 3*std,  "valeur de l'overlap p ", overlap_p)
                        print("_"*20)
                else:
                    if verbose:
                        print("anchor :\n", sentence_candidat[0:100])
                        print("a pour doublon SANS OVERLAP:", sentence_candidat_doublon[0:100])
                        print("Valeur de la cosine ", top_candidat[0], " et la valeur du threshold ", nu + 3*std, "valeur de l'overlap p ", overlap_p)
                        print("_"*20)
        else:
            if verbose:
                print("no doublon :", sentence_candidat)
                print("-"*20)

    return positions_doublons, doublons, o

def manage_duplicate(sentences:List[str], max_df=ttsv.MAX_DF, min_df=ttsv.MIN_DF, verbose=ttsv.VERBOSE, MINIMAL_CHUNK_SIZE = ttsv.MINIMAL_CHUNK_SIZE) -> Tuple[List[str], List[str], List[int], Dict[str,str], Any]:
    """
    Cette fonction gère les duplicats rapport par rapport Et retire les doublons de sentences.

    Explication des différentes variables : 
    sentences_minimum:List[str] = [] # Dans cette liste on va mettre uniquement le contenu de l'écart/remarque, sans le contexte. Pour pouvoir comparer les doublons proprement.
    flat_sentences_min # Une version applati de sentences_minimum

    map_sent_positions : permet de faire le mapping entre la "minimum sentence" extraite et sa position dans l'ordre des sentences (chunk_complet).
    """
    # Préparation pour TFIDF
    final_stopwords_list = stopwords.words('french')
    vectorizer_ = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=(1, 1), stop_words=final_stopwords_list)
    
    # Préparation des objets 
    SentencesMinimum = Annotated[List[str], "Dans cette liste on va mettre uniquement le contenu de l'écart/remarque, sans le contexte. Pour pouvoir comparer les doublons proprement."]
    sentences_minimum: SentencesMinimum = []

    # sentences_minimum:List[str] = [] # Dans cette liste on va mettre uniquement le contenu de l'écart/remarque, sans le contexte. Pour pouvoir comparer les doublons proprement.
    
    MapSentencesPositions = Annotated[Dict[str, int], "Dictionnaire qui permet le mapping entre un écart ou doublon et sa position dans la liste avec contexte."]
    map_sent_positions : MapSentencesPositions = dict() # Dans le cas où il y a plusieurs 
    for idx, text in enumerate(sentences): # On récupère les chunks un par un
        res = split_text(text)
        sentences_minimum.append(res)
        for _ in res:
            map_sent_positions[_] = idx 
    flat_sentences_min = list(reduce(lambda x,y: x+y,sentences_minimum, [])) # flatten liste
    flat_sentences_min = [x for x in flat_sentences_min if len(x)>MINIMAL_CHUNK_SIZE]

    # Gestion des doublons
    positions_doublons, doublons, cosine = spot_doublons(flat_sentences_min, vectorizer=vectorizer_, verbose=verbose)
    unduplicated_flat_sentences_min= [i for j, i in enumerate(flat_sentences_min) if j not in positions_doublons] # On a donc les sentences minimum non duppliqué, maintenant il faut les relier aux chuunks complet (sentences)

    UnduplicatedChunksMaxContext = Annotated[List[str], "Cette liste stockera les chunks avec contexte. Et en cas de doublon celui qui a le + grand contexte."]
    unduplicated_chunks_max_context: UnduplicatedChunksMaxContext = []

    for min_chunk in unduplicated_flat_sentences_min:
        # Cette boucle permet de récupérer le chunk qui a le + de contexte associé à un chunk minimaliste (donc uniquemnt l'écart ou la remarque)

        chunk = sentences[map_sent_positions[min_chunk]] # On récupère le chunk analysé avec tout son contexte.
        # Sauf que ce n'est pas forcément le chunk le plus pertinent, peut être que son doublon contient + de contexte. 
        is_doublon = doublons.get(min_chunk, False) # False si jamais le chunk analysé n'a pas de doublon.
        if is_doublon: 
            doublons_chunk = sentences[map_sent_positions[is_doublon]]
            if len(chunk) >= len(doublons_chunk):
                unduplicated_chunks_max_context.append(chunk)
            else:
                unduplicated_chunks_max_context.append(doublons_chunk)
        else:
            unduplicated_chunks_max_context.append(chunk)
    
    return unduplicated_chunks_max_context, flat_sentences_min, positions_doublons, doublons, cosine