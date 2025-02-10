import json
import importlib.resources
import re
from typing import Deque, List, Optional, Tuple, Any
import unicodedata

import pandas as pd
from docx import Document
from docx.document import Document as _Document
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from docx.table import _Cell, Table, _Row
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from pandas.io import excel


from text_extraction import settings_variables as sv

WORDS_TO_REMOVE = sv.WORDS_TO_REMOVE
REGEX_FLAG = sv.MOTIF_RE_SOUPLE_BUT_AT_THE_BEGINNING_OF_SEQUENCES

def nettoyer_chaine(chaine:str) ->str:
    """
    Cette fonction nettoie chaine de caractère non imrprimable.
    """
    # Utilise une expression régulière pour supprimer les caractères non imprimables
    chaine_propre = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]|<0x[0-9A-Fa-f]+>', '', chaine)
    return chaine_propre

def grant_encoding(msg:str) -> str:
    return unicodedata.normalize("NFKD", msg)

def remove_words(msg:str, list_words_to_remove = WORDS_TO_REMOVE)->str:
    """
    Cette fonction permet de retirer du texte les mots présents dans list_words_to_remove
    """
    for mot in list_words_to_remove:
        # Utilisation d'une expression régulière pour retirer le mot suivi d'une virgule ou d'un saut de ligne
        pattern = re.compile(rf'{mot}(?:,|\n)?', re.IGNORECASE) # Comme le REGEX Change à chaque fois, on l'externalise pas.
        msg = pattern.sub("", msg)
    return msg


def load_acronyme_data(glossaire_path):
    with importlib.resources.open_text('text_extraction', glossaire_path, encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data

def remplacer_acronymes(chaine:str, glossaire_path=r"acronyme_mapping.json"):
    """
    Cette fonction permet de remplacer dans la chaine d'entrée les acronymes présetns dans le fichier JSON au chemin glossaire_path.
    """
    glossaire = None
    if glossaire_path.endswith("json"): # Si JSON
        glossaire = load_acronyme_data(glossaire_path)
    if glossaire:
        for acronyme, definition in glossaire.items():
            word_chars = r"\w'’\"«»“”"
            pattern = re.compile(r'(?<![' + word_chars + r'])' + re.escape(acronyme) + r'(?![' + word_chars + r'])', re.IGNORECASE)
            chaine = pattern.sub(definition, chaine)
        return chaine
    else:
        raise ValueError("Le glossaire des acronymes n'a pas pu être instancié.")
        return chaine

def remplacer_nlines(chaine:str)->None:
    pass
    # Au début je voulais l'implémenter mais en fait il y a certains tableaux qui ont le nombre de saut de lignes nécessaires pour faire un truc visuel. A voir si je le garde ou pas.

def preprocess(chaine:str)->str:
    chaine = nettoyer_chaine(chaine)
    
    chaine = remplacer_acronymes(chaine)
    return chaine

def dict_to_markdown(table_dict):
    headers = table_dict.get('headers', [])
    data = table_dict.get('data', [])

    # Déterminer le nombre de colonnes
    num_cols = len(headers)

    # Remplacer les en-têtes vides par des espaces pour éviter les problèmes de formatage
    headers = [header if header else ' ' for header in headers]

    # Créer la ligne des en-têtes
    header_line = '| ' + ' | '.join(headers) + ' |'

    # Créer la ligne de séparation
    separator_line = '| ' + ' | '.join(['---'] * num_cols) + ' |'

    # Fonction pour échapper les caractères spéciaux Markdown
    def escape_markdown(text):
        if not isinstance(text, str):
            text = str(text)
        # Échapper les pipes et autres caractères si nécessaire
        text = text.replace('|', '\\|').replace('\n', '<br>')
        return text

    # Traiter les données
    data_lines = []
    for row in data:
        # S'assurer que chaque ligne a le même nombre de colonnes
        row = list(row)
        if len(row) < num_cols:
            row += [''] * (num_cols - len(row))
        elif len(row) > num_cols:
            row = row[:num_cols]
        # Échapper les contenus des cellules
        escaped_row = [escape_markdown(cell) for cell in row]
        data_line = '| ' + ' | '.join(escaped_row) + ' |'
        data_lines.append(data_line)

    # Assembler toutes les lignes
    markdown_table = '\n'.join([header_line, separator_line] + data_lines)
    return markdown_table

def liste_vers_dataframe(ma_liste, plain_text_column="plain_text"):
    """
    Convertir la liste des lignes des Docx en dataframe.
    """
    # Liste pour stocker les lignes du DataFrame
    lignes = []

    for element in ma_liste:
        if isinstance(element, dict):
            # Si l'élément est un dictionnaire, on l'ajoute tel quel
            lignes.append(element)
        elif isinstance(element, str):
            # Si l'élément est une chaîne, on crée un dictionnaire 
            lignes.append({plain_text_column: element})
        else:
            # Gérer d'autres types si nécessaire
            lignes.append({plain_text_column: str(element)})

    # Créer le DataFrame à partir des lignes
    df = pd.DataFrame(lignes)
    return df

def contient_motif(texte:list|str, motif=REGEX_FLAG):
    # Compile le motif regex une seule fois
    motif = re.compile(motif, flags=re.IGNORECASE)
    
    if isinstance(texte, str):
        # Si 'texte' est une chaîne unique, vérifie si le motif est présent
        return bool(motif.search(texte))
    elif isinstance(texte, list):
        # Si 'texte' est une liste, vérifie chaque élément
        return any(motif.search(s) for s in texte if isinstance(s, str))
    else:
        # Si 'texte' n'est ni une chaîne ni une liste, retourne False
        return False

def extract_paragraph_style_not_normal(paragraph:Paragraph, normal_style_str="Normal") -> str|None:
    """
    Cette fonction regarde si le style appliqué au paragraph n'est pas un style "normal" pour récupérer ce style en tant que texte.
    Pourquoi ? Par exemple le fichier 240913_R_EHPAD_PECM.docx a un style "Ecart" et "Remarque" qui ne remontent pas lors de l'extraction de texte.
    Ce n'est pas le seul document concerné je suppose.
    """
    if paragraph.style.name != normal_style_str:
        return paragraph.style.name
    return ""

def keep_first_in_list(liste:list, filler="")->list:
    """
    Cette fonction analyse liste et si il y a plusieurs occurences identiques qui se suivent alors la liste retournée ne garde que la première les autres occurences sont remplacés par filler.
    Pourquoi dans notre cas ? Car python-docx a une manière spécifique de gérer les matrices qui amènent des répétitions. https://python-docx.readthedocs.io/en/latest/user/tables.html 
    """
    return [liste[i] if i == 0 or liste[i] != liste[i - 1] else filler for i in range(len(liste))]

def is_header_row(row):
    trPr = row._tr.trPr
    if trPr is not None:
        tblHeader = trPr.find(qn('w:tblHeader'))
        return tblHeader is not None
    return False

def is_nested_list(l):
    try:
          next(x for x in l if isinstance(x,list))
    
    except StopIteration:
        return False
    
    return True



def paragraph_to_markdown(paragraph: Paragraph) -> str:
    """
    Convertit un objet paragraph de python-docx en une chaîne Markdown.
    Gère les titres en fonction des styles de paragraphe.
    """
    text = paragraph.text.strip()
    if not text:
        return ""
    
    # Mapping des styles de titre Word aux niveaux Markdown
    style = paragraph.style.name.lower()
    if 'titre 1' in style:
        return f"# {text}"
    elif 'titre 2' in style:
        return f"## {text}"
    elif 'titre 3' in style:
        return f"### {text}"
    # Ajouter plus de niveaux si nécessaire
    else:
        return text

def table_to_markdown(table: Table) -> str:
    """
    Convertit un objet table de python-docx en une chaîne Markdown.
    """
    markdown = []
    
    # Extraction des en-têtes
    headers = [cell.text.strip() for cell in table.rows[0].cells]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(['---'] * len(headers)) + " |"
    markdown.append(header_line)
    markdown.append(separator_line)
    
    # Extraction des lignes de données
    for row_idx, row in enumerate(table.rows[1:], start=2):
        row_data = [cell.text.strip() for cell in row.cells]
        row_line = "| " + " | ".join(row_data) + " |"
        markdown.append(row_line)
    
    return "\n".join(markdown)

def iter_block_items(parent:Any):
    """
    Fonction pour parcourir l'arbre docx
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    elif isinstance(parent, _Row):
        parent_elm = parent._tr
    else:
        raise ValueError("Something is not right")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def process_paragraph(paragraph:Paragraph, chapter:str, subchapter:str, empty_line = "") -> Tuple[dict|bool, str, str] :
        """Fonction qui applique les traitements à un objet Paragraph"""
        text:str = paragraph.text.strip()
        if text != empty_line : 
            flag = contient_motif(text)
            para_info = {
            "text": preprocess(text),
            "style": paragraph.style.name,
            "flag" : flag
                }
            
            if paragraph.style.name == 'Heading 1': # Chapter et subchapter étant init je ne garde que la valeur précédente en mémoire (principe d'une hiérarchie)
                chapter = text
            elif paragraph.style.name == 'Heading 2':
                subchapter = text
            return para_info, chapter, subchapter

        else:
           return False, chapter, subchapter

def process_table(table:Table)->Tuple[list, list, list]:
    """Fonction qui applique les traitements à un objet Table"""
    markdown_table = table_to_markdown(table)
    table_data:List[Any] = []
    headers:List[str] = []
    flags = []
    has_header:bool = False
    for idx, row in enumerate(table.rows):
        row_data_loop = []
        for cell in row.cells:
            previous_style = None
            para_lines = []
            for para in cell.paragraphs:
                prefix = extract_paragraph_style_not_normal(para)
                if previous_style != prefix:
                    para_lines.append(prefix + " " + preprocess(para.text))
                else:
                    para_lines.append(preprocess(para.text))
                previous_style = prefix
                
            row_data_loop.append("\n".join(para_lines))
        processed_row_data_loop = keep_first_in_list(row_data_loop)

        if is_header_row(row):
            headers = processed_row_data_loop
            has_header = True
        else:
            table_data.append(processed_row_data_loop) # Chaque élément de table data correspond à une ligne.
    # Si aucune ligne d'en-tête n'est détectée, on suppose que la première ligne est l'en-tête
    if not has_header and table_data:
        headers = table_data.pop(0)
    assert not is_nested_list(headers), "headers is a nested List. Should be a list of str"
    flag_header = contient_motif(headers)
    flags.append(flag_header)

    for values in table_data:
        # Pour avoir un flag au niveau de la ligne.
        flag = contient_motif(values)
        flags.append(flag)
        #Pour avoir un flag au niveau de la cellule : script ci après.
        #for value in values:
        #    flag = contient_motif(value)
        #    flags.append(flag)
    return table_data, headers, flags

def concat_listes_df_and_save_on_excel(dfs: list[pd.DataFrame], excel_name:str) ->bool:
    """
    Cette fonction concatène une liste de Dataframe et sauvegarde la sortie dans un excel.
    """
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    excel_sufix = ".xlsx"
    if excel_name.endswith(excel_sufix):
        df.to_excel(excel_name)
    else:
        df.to_excel(excel_name+excel_sufix)
    return True

def concat_listes_df_and_save_on_pickle(dfs: list[pd.DataFrame], pkl_name:str) ->bool:
    """
    Cette fonction concatène une liste de Dataframe et sauvegarde la sortie dans un excel.
    """
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    excel_sufix = ".pkl"
    if pkl_name.endswith(excel_sufix):
        df.to_pickle(pkl_name)
    else:
        df.to_pickle(pkl_name+excel_sufix)
    return True

def extract_word_info(file_path):
    """
    Fonction pour extraire toutes les informations possibles d'un fichier Word.
    """
    document = Document(file_path)

    # Stockage des résultats
    result = {
        "paragraphs": [],
        "tables": [],
        "headings": [],
        "hyperlinks": [],
        "metadata": {},
        "content_raw" : [],
        "content_markdown" : []
    }

    # Extraction des métadonnées
    core_properties = document.core_properties
    result["metadata"] = {
        "title": core_properties.title,
        "author": core_properties.author,
        "created": core_properties.created,
        "modified": core_properties.modified,
        "subject": core_properties.subject,
        "keywords": core_properties.keywords,
    }
    
    
    # Initialisation des variables pour hiérarchiser. Pour le moment chapter et subchapter uniquement
    chapter:str = ""
    subchapter:str = ""
    
    for block in iter_block_items(document): # Pour chaqu bloc de document
        if isinstance(block, Paragraph):
            para = block
            
            para_info, chapter, subchapter = process_paragraph(para, chapter, subchapter)
            if para_info:
                result["content_raw"].append(para_info)

                result["headings"].append((chapter, subchapter))
        elif isinstance(block, Table):
            table = block
            table_data, headers, flags = process_table(table)

            result["tables"].append({"headers": headers, "data": table_data})
            result["content_raw"].append({"headers": headers, "data": table_data, "flag":flags})
            result["headings"].append((chapter, subchapter))

    return result