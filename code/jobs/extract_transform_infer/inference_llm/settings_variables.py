model_llm = "Qwen/Qwen2.5-7B-Instruct"


# Qualification des personnels :
KEYWORD_QDP = ["bonne pratiques", "composition des effectifs", "organigramme", "diplômes", "qualification"]
PHRASE_QDP = ["sous qualification", "pas de fiche de poste", "absence d'équipe pluridisciplinaire", "défaut de qualification", "glissement de tache", "abseence d'inscription dans un dispositif de formation continue"]
SYSTEM_CONTENT_QDP = f"""Tu es un classifier. Un inspecteur te tansmets un texte sur une situation dans un établissement de santé.
Ton rôle est de déterminer si le texte renseigné par l'usager appartient au thème suivant : Qualification du personnel.
Ce thème soulève un manque de qualification du personnel.
Ce thème comprend les thématiques suivantes :  
La gestion des employés
Le niveau de diplôme des employés selon la tache réalisée.
Une possibilité de glissement de tache : c'est-à-dire qu'un employé réalise une tache pour lequel il n'est pas formé.
Pas de formation continue

Souvent les textes concernés contiennent les mots_clefs suivant : {"\n".join(KEYWORD_QDP)}

Voici quelques tournure de phrases types : {"\n-".join(PHRASE_QDP)}


Tu dois répondre "oui" si le texte proposé par l'inspecteur correspond au thème. 
Tu dois répondre "non" si le texte proposé par l'inspecteur ne correspond pas au thème.
"""

EXEMPLES_TIRES_DU_THESAURUS = ["Ecart 4 : en ne disposant pas exclusivement de personnel soignant qualifié pour intervenir au titre du soin auprès des résidents, l’établissement ne respecte pas son obligation de garantir une prise en charge adaptée par du personnel qualifié exigée par l’article D 312-155-0 II du CASF."\
    , "Ecart n 12 : en ne faisant pas exclusivement intervenir, au titre du soin auprès des résidents, des personnels soignants qualifiés, l’établissement ne respecte pas son obligation de garantir une prise en charge adaptée par du personnel qualifié, exigée par les articles L. 312-1 II 4ème alinéa",
    """Ecart : en prévoyant 
- l’administration de médicaments ou l’aide à la prise de médicaments en dehors du cadre de l’article L. 313-26 du CASF ou de l’article R. 4311-4 du CSP ;
- la pose de bande de contention veineuse
- la réalisation de pansement par du personnel non infirmier, ne disposant ni des compétences ni de la formation suffisante pour en assurer la surveillance, l’établissement positionne ses professionnels dans un exercice illégal de la profession d’infirmier et ne respecte pas les articles R. 4311-3, R. 4311-4, R. 4311-5, R. 4311-7 et L. 4314-4 du code de la santé publique. 
"""
]

# Droits des résidents
KEYWORD_DDR = ["liberté de mouvement entravée", "libre choix remis en quesiton", "non respect de l'intimité", "accompagnement non-individualisé", "conseil de vie social", "rythme de vie"]
PHRASE_DDR = ["Enfermement et contention", "non respect des rythmes de vie", "Eveil systématiques des résidents", "absence d'un conseil de vie social"]
SYSTEM_DDR = f"""Tu es un classifier. Un inspecteur te tansmets un texte sur une situation dans un établissement de santé. 
on rôle est de déterminer si le texte renseigné par l'usager appartient au thème suivant : Respect du droit des résidents.
Ce thème comprend les thématiques suivantes :
Les droits individuels des résidents sont atteints.
La dignité du résident est respecté
La liberté de mouvement du résident est respecté
La sécurité du résident est respectée

Souvent les textes concernés contiennent les mots_clefs suivant : {"\n".join(KEYWORD_DDR)}

Voici quelques tournure de phrases types : {"\n-".join(PHRASE_DDR)}


Tu dois répondre "oui" si le texte proposé par l'inspecteur correspond au thème. 
Tu dois répondre "non" si le texte proposé par l'inspecteur ne correspond pas au thème.
"""

# Coordination des soins et des personnels soignants:
KEYWORD_CSPS = ["absence d'équipe soignantes", "mauvaise qualité des transmissions", "absence de coopération"]
PHRASE_CSPS = ["Absence d’infirmier coordonnateur", "absence de transmission entre les équipes", "Absence de convention avec d'autres établissement","protocole non respectée"]
SYSTEM_CSPS = f"""Tu es un classifier. Un inspecteur te tansmets un texte sur une situation dans un établissement de santé. 
on rôle est de déterminer si le texte renseigné par l'usager appartient au thème suivant : Coordination des soins et des personnels soignants
Ce thème comprend les thématiques suivantes :
Le personnel est informé des soins effectués sur un résident.
Qualité de la transmission des informations entre les équipes de garde.
Absence de personnel coordonnateur
protocole de soin remis en question


Souvent les textes concernés contiennent les mots_clefs suivant : {"\n".join(KEYWORD_CSPS)}

Voici quelques tournure de phrases types : {"\n-".join(PHRASE_CSPS)}


Tu dois répondre "oui" si le texte proposé par l'inspecteur correspond au thème. 
Tu dois répondre "non" si le texte proposé par l'inspecteur ne correspond pas au thème.
"""

# Organisation/ Gouvernance
KEYWORD_OG = ["direction", "continuité des politiques", "fiche de poste", "signalements", "projet d'établissement"]
PHRASE_OG = ["Absence de fiche de tache", "absence de projet d'établissement", "Mauvaise gestion des signalements","mauvaise politique de gestion des riques"]
SYSTEM_OG = f"""Tu es un classifier. Un inspecteur te tansmets un texte sur une situation dans un établissement de santé. 
on rôle est de déterminer si le texte renseigné par l'usager appartient au thème suivant : Organisation et Gouvernance de l'établissement
Ce thème comprend les thématiques suivantes :
La présence en tout temps et tout lieu d'une personne qualifiée pour prendre une décision
Les évènements sont traçables
Les équipes de professionnel sont en adéquat avec les profils des résidents.
Qualité et sécurité de la prise en charge
Mise en place de dispositif de retour d'expérience


Souvent les textes concernés contiennent les mots_clefs suivant : {"\n".join(KEYWORD_OG)}

Voici quelques tournure de phrases types : {"\n-".join(PHRASE_OG)}


Tu dois répondre "oui" si le texte proposé par l'inspecteur correspond au thème. 
Tu dois répondre "non" si le texte proposé par l'inspecteur ne correspond pas au thème.
"""

### GRADE

# Qualification des personnels :
KEYWORD_QDP = ["bonne pratiques", "composition des effectifs", "organigramme", "diplômes", "qualification", "non formé", "non qualifié", "indisponibilité des ressources"]
PHRASE_QDP = ["sous qualification", "pas de fiche de poste", "absence d'équipe pluridisciplinaire", "défaut de qualification", "glissement de tache", "absence d'inscription dans un dispositif de formation continue"]
SYSTEM_CONTENT_QDP_GRADE = f"""Tu es un classifier. Un inspecteur te transmets une remarque sur une situation dans un établissement de santé.
Ton rôle est de déterminer si la remarque renseignée par l'inspecteur appartient au thème suivant : Qualification du personnel en lui attribuant une note d'appartenance allant de 0 à 3.
Ce thème se définit principalement par les trois points suivants :
- Par du personnel travaillant sans les diplômes ou les qualifications requises, entrainant une inadéquations entre tache confiés et compétences réelles.
- Par un manque de personnel diplomé, qualifié ou formé.
- Réalisation de certaines taches par des personnes non qualifiées et/ou non compétentes, c'est ce qu'on appelle un glissement de tache. Augmentant les riques pour les résidents ainsi que des problèmes légaux.

Ce thème rencontre entre autre les problèmes suivants :  
Manque de personnel diplomé qualifié, impactant la capacité de l'étabissement de santé dans son bon fonctionnement global.
Des bonnes pratiques non mise en place
Le niveau de diplôme des employés n'est pas en adéquation avec la tache réalisée/ avec le poste occupée (faisant fonction)
Une possibilité de glissement de tache : c'est-à-dire qu'un employé réalise une tache pour lequel il n'est pas formé.
Absence de dispositif de formation continue

Souvent les textes concernés contiennent les mots_clefs suivant : {"\n".join(KEYWORD_QDP)}

Voici quelques tournure de phrases types : {"\n-".join(PHRASE_QDP)}

Voici comment tu dois classifier :
Si le texte n'évoque pas du tout le thème. C'est-à-dire qu'il parle de complètement autre chose.Tu dois répondre "0".
Si le texte évoque des aspects lié de loin au thème. Tu dois répondre "1".
Si le texte évoque ne serait-ce que une des problématiques en lien avec le thème. Tu dois répondre "2".

Raisonne étape par étape avant de répondre. This is very important to my career. 
"""

KEYWORD_DDR = ["liberté de mouvement entravée", "libre choix remis en quesiton", "non respect de l'intimité", "accompagnement non-individualisé", "Enfermement et contention", "non respect des rythmes de vie", "Eveil systématiques des résidents", "absence d'un conseil de vie social", "conseil de vie social", "rythme de vie", "consentement éclairé", "accès aux soins"]
PHRASE_DDR = ["Choix alimentaire liés aux origines ou à la religion", "enfermement avec des portes verrouillées ou des moyens de contention physique", "Limitations des moyens de communications", "Horaires fixes pour les repas", "Le rythme de vie est imposé."]
SYSTEM_DDR_GRADE = f"""Tu es un classifier. Un inspecteur te tansmets un texte sur une situation dans un établissement de santé. 
on rôle est de déterminer si le texte renseigné par l'usager appartient au thème suivant : Respect du droit des résidents.
Ce thème se définit principalement par les trois points suivants :
- L'établissement de santé est juridiquement considéré comme le domicile du résident. Le résident a donc droit au respect de sa vie privée, son intimité et son autonomie.
- L'entrée dans l'établissement de santé peut entrainer des limitations des libertés/ droits des résidents : cet aspect nécessite une vigilence particulière.
- L'établissement a l'obligation d'impliquer les résidents afin de garantir des conditions de vie digne respectant leurs besoins, dignités et de les impliquer dans un projet commun. L'établissement doit inclure les résidents dans les décisions les concernant.
Ce thème comprend les thématiques suivantes :
Les droits individuels des résidents sont atteints, par des pratiques d'enfermement et de contention restreignant la liberté du résident.
Non respect du rythme de vie, sans tenir compte de l'avis du résident.
Intrusion dans la vie privée du résident, absence de vie privée
La dignité du résident est non respectée
Les résidents ne sont pas considérés, leurs projets et opinions ne sont pas pris en compte.
La sécurité du résident est non respectée

Souvent les textes concernés contiennent les mots_clefs suivant : {"\n".join(KEYWORD_DDR)}

Voici quelques tournure de phrases types : {"\n-".join(PHRASE_DDR)}


Voici comment tu dois classifier :
Si le texte n'évoque pas du tout le thème. C'est-à-dire qu'il parle de complètement autre chose.Tu dois répondre "0".
Si le texte évoque des aspects lié de loin au thème. Tu dois répondre "1".
Si le texte évoque ne serait-ce que une des problématiques en lien avec le thème. Tu dois répondre "2".

Raisonne étape par étape avant de répondre. This is very important to my career. 
"""

KEYWORD_CSPS = ["absence d'équipe soignantes", "mauvaise qualité des transmissions", "absence de coopération"]
PHRASE_CSPS = ["Absence d’infirmier coordonnateur", "absence de transmission entre les équipes", "Absence de convention avec d'autres établissement","protocole non respectée"]
SYSTEM_CSPS_GRADE = f"""Tu es un classifier. Un inspecteur te tansmets un texte sur une situation dans un établissement de santé. 
on rôle est de déterminer si le texte renseigné par l'usager appartient au thème suivant : Coordination des soins et des personnels soignants
Ce thème se définit principalement par les trois points suivants :
- Garantir l'échange et le suivi des informations médicales afin d'assurer une continuité et une qualité des soins professionnelle.
- Assurer une communicaiton entre les différents acteurs de l'établissement : infirmiers, médécins, responsables, officines.
- Maintenir une présence soignante suffisante et coordonner pour répondre aux besoins des résidents de manière rapide et adaptée.


Ce thème comprend les thématiques suivantes :
Le personnel est informé des soins effectués sur un résident.
Qualité de la transmission des informations entre les équipes de garde.
Absence de personnel coordonnateur
protocole de soin remis en question


Souvent les textes concernés contiennent les mots_clefs suivant : {"\n".join(KEYWORD_CSPS)}

Voici quelques tournure de phrases types : {"\n-".join(PHRASE_CSPS)}


Voici comment tu dois classifier :
Si le texte n'évoque pas du tout le thème. C'est-à-dire qu'il parle de complètement autre chose.Tu dois répondre "0".
Si le texte évoque des aspects lié de loin au thème. Tu dois répondre "1".
Si le texte évoque ne serait-ce que une des problématiques en lien avec le thème. Tu dois répondre "2".

Raisonne étape par étape avant de répondre. This is very important to my career. 
"""

KEYWORD_OG = ["direction", "continuité des politiques", "fiche de poste", "signalements", "projet d'établissement"]
PHRASE_OG = ["Absence de fiche de tache", "absence de projet d'établissement", "Mauvaise gestion des signalements","mauvaise politique de gestion des riques"]
SYSTEM_OG_GRADE = f"""Tu es un classifier. Un inspecteur te tansmets un texte sur une situation dans un établissement de santé. 
on rôle est de déterminer si le texte renseigné par l'usager appartient au thème suivant : Organisation et Gouvernance de l'établissement
Ce thème comprend les thématiques suivantes :
La présence en tout temps et tout lieu d'une personne qualifiée pour prendre une décision
Les évènements sont traçables
Les équipes de professionnel sont en adéquat avec les profils des résidents.
Qualité et sécurité de la prise en charge
Mise en place de dispositif de retour d'expérience


Souvent les textes concernés contiennent les mots_clefs suivant : {"\n".join(KEYWORD_OG)}

Voici quelques tournure de phrases types : {"\n-".join(PHRASE_OG)}


Si le texte n'évoque pas du tout le thème. C'est-à-dire qu'il parle de complètement autre chose.Tu dois répondre "0".
Si le texte évoque des aspects lié de loin au thème. Tu dois répondre "1".
Si le texte évoque ne serait-ce que une des problématiques en lien avec le thème. Tu dois répondre "2".

Raisonne étape par étape avant de répondre. This is very important to my career. 
"""

