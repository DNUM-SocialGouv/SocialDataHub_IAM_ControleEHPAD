from typing import Dict, List, Tuple, Any
import json
import importlib.resources

import matplotlib.pyplot as plt

from openai import OpenAI
import openai


def get_client_openai(params:Dict) :
    client = OpenAI(
    **params
    )
    return client



def get_settings_params(params_path="settings_llm.json"):
    with importlib.resources.open_text('inference_llm', params_path, encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data

def do_inference(client,  messages:List[Dict], llm_params:Dict, model_llm:str, extra_body:Dict|None=None):
    """
    Cette fonction permet de faire une inférence au client indiqué.

    llm_params : les paramètres utilisés par le standard OpenAI API
    messages : List des messages au standard OpenAI
    client : client standard OpenAI
    model_llm : nom du modèle
    extra_body : Pour demander des informations transmises par la surcouche VLLM mais pas par standard OpenAI.
    """
    completion = client.chat.completions.create(
        model = model_llm, messages=messages,
        **llm_params,
        extra_body=extra_body

    )
    return completion

def get_colors_from_values(value, vmin=0, vmax=1, cmap=plt.cm.viridis):
    """Retourne le code couleur associé à la value"""
    norm=mcolors.Normalize(vmin=vmin, vmax=vmax)

    return cmap(norm(value))

def construct_beam_graph(result:openai.types.chat.chat_completion.Choice, beam_number:int):
    """Retourne le graph d'un retour openAi avec des logprobs. Ainsi que les couleurs de chaque noeud (pour la visualisation)
    beam_number influe directement sur la représentation, merci de l'incrémenter à chaque fois.
    """ 
    G = nx.DiGraph()
    #pos = {}
    tok_strs = []
    colors = []
    for token in result.logprobs.content:
        prob = round(np.exp(token.logprob), 4)
        tok = token.token
        tok_str = f"[{abs(beam_number)}]\n{tok}" + "\n("+str(prob)+")"
        tok_strs.append(tok_str)
        colors.append(get_colors_from_values(prob))
    assert len(colors) == len(tok_strs)
    G.add_nodes_from(tok_strs)
    for i in range(len(tok_strs) - 1):
        G.add_edge(tok_strs[i], tok_strs[i + 1])
        couleurs = colors[i]
        G.nodes[tok_strs[i]]["couleur"] = couleurs
        G.nodes[tok_strs[i]]["pos"] = (i, beam_number)
    
    # Dernier élément de boucle
    couleur = colors[-1]
    G.nodes[tok_strs[-1]]["couleur"] = couleur
    G.nodes[tok_strs[-1]]["pos"]  = (len(tok_strs), beam_number)

    return G

def construct_beam_graphs(results:list, display=True, font_size=10, node_size=1000):
    """Affiche le graph des beams search.
    
    
    Fonction en pause, si jamais y'a des tokens redondants ça va se superposer, c'est pas très malin."""
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    G = nx.DiGraph()
    positions = {}
    couleurs_noeuds = []
    for idx_resutl, result in enumerate(results):
        print(idx_resutl, result)
        Gsub = construct_beam_graph(result, -idx_resutl)
        pos = { mot : Gsub.nodes[mot]["pos"] for mot in Gsub.nodes()
    
            }
        couleurs_noeuds += [Gsub.nodes[mot]['couleur'] for mot in Gsub.nodes()]
        positions.update(pos)
        G = nx.compose(G, Gsub)
    if display :
        nx.draw_networkx(G, pos=positions, node_color=couleurs_noeuds, 
        font_size=font_size, node_size=node_size, ax=ax)
        cbar = plt.colorbar(sm, ax=ax)
        plt.tight_layout()
    else:
        return G