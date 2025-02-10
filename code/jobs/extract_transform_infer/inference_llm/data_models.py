# Format
from typing import List, Tuple, Any, Dict
import json
from pydantic import BaseModel, Field,  root_validator, ValidationError
from enum import Enum
# Externe
import numpy as np
import openai
import torch.nn as nn
import torch

NEGATIV_TOKENS = ["non", " non", "no", "_no", "NO"]
POSITIV_TOKENS = ["oui", " oui", "yes", "YES"]

class Grade(str, Enum):
    zero  ="0"
    un ="1"
    deux = "2"

class Raisonnement(BaseModel):
    raisonnement:str
    verdict:Grade

class Binary_results(BaseModel):
    answers : List[openai.types.chat.chat_completion.Choice]
    positiv_probs: float = None
    negativ_probs: float = None

    def get_classif_weights(self) -> Tuple[float, float]:
        """
        Pour une réponse donnée du tyle "réponds oui ou non", retourne la proba du oui et du non.
        """
        result = self.answers[0] # Pour le moment je prends UNIQUEMENT le top beam.
        logprobs: List = result.logprobs.content[0].top_logprobs
        negativ_probs: float = 0
        positiv_probs: float = 0
        for logprob in logprobs:
            token = logprob.token
            prob = np.exp(logprob.logprob)
            if token in NEGATIV_TOKENS:
                negativ_probs+=prob
            if token in POSITIV_TOKENS:
                positiv_probs += prob
        self.positiv_probs = positiv_probs
        self.negativ_probs = negativ_probs
        return positiv_probs, negativ_probs
    
    
    def model_post_init(self, __context: Any) -> None:
        self.get_classif_weights()

class Grade_Results(BaseModel):
    answers : List[openai.types.chat.chat_completion.Choice]
    confidences:List[Tuple[int, float]] = None
    distribution_function : Any = nn.Softmax(dim=0)
    distributions : Any = None


    def compute_confidence(self, kw_note:str="verdict")->List[Tuple[int, float]]:
        """
        Pour une réponse où on force le LLM à raisonner et donner une note selon un JSON. On compute la proba de la génération et on la pondère par la note qu'il donne sur ses "n" meilleures générations.
        """
        confidences = []
        for answer in self.answers:
            mean_logprob = sum([x.logprob for x in answer.logprobs.content])/len(answer.logprobs.content) # meaned perplexity pour la réponse.
            confidence = np.exp(mean_logprob)

            json_answer = json.loads(answer.message.content)
            verdict = json_answer[kw_note]
            confidences.append((verdict, confidence))
        self.confidences = confidences

        return confidences
    
    def voting(self, classes=[0, 1,2 ],eps = 0.0002):
        """
        A voir pour la suite comment on calera automatiquement les ranges de classe sur Grade.
        Pour le moment le voting est simplement une moyenne pondéré et le max l'emporte et on fournit une fiabilité de prédiction.
        """
        #self.distribution_function = nn.Softmax(dim=0)
        confidence_grade = []
        for classe in classes:
            confidence_grade.append(sum([x[1] for x in self.confidences if int(x[0])==classe])+eps)
        self.distributions = self.distribution_function(torch.tensor(confidence_grade))

        return self.distributions

    def get_grade_and_answers(self):
        appartenance_0_1_2 = tuple(map(lambda x: round(x, 3)*100, self.distributions.numpy()))
        llm_answers = [x.message.content for x in self.answers]

        return appartenance_0_1_2, llm_answers
        
    def model_post_init(self, __context: Any) -> None:
        self.compute_confidence()
        self.voting()


class QDP_results(Binary_results):
    answers : List[openai.types.chat.chat_completion.Choice]

class DDR_results(Binary_results):
    answers : List[openai.types.chat.chat_completion.Choice]

class CSPS_results(Binary_results):
    answers : List[openai.types.chat.chat_completion.Choice]

class OG_results(Binary_results):
    answers : List[openai.types.chat.chat_completion.Choice]


class Result(BaseModel):
    sentence: str # Chunk à analyser.
    prompts : Dict[str, str] = Field(required=True,description="Dictionnaire des prompts, en clef l'acronyme pour identifier le prompt et en valeur, le prompt système pour la classif")
    qdp: QDP_results | None = None 
    ddr: DDR_results | None = None
    csps: CSPS_results | None = None
    og: OG_results | None = None

    @root_validator(pre=True)
    def check_required_keys(cls, values):
        # Vérifie que le dico des promtps contient bien les 4 prompts annoncés.
        if "prompts" in values:
            required_keys = {"QDP", "DR", "CSPS", "OG"} 
            # Qualification du personnel, Droits des Residents, 
            # Coordination des soins et des personnels soignants, Organisation/ Gouvernance
            provided_keys = set(values["my_dict"].keys())
            missing_keys = required_keys - provided_keys
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}")
        return values



