WORDS_TO_REMOVE = []
MOTIF_RE_LACHE = r"(\bécart\b|\becart\b|\bremarque\b|\be\b|\br\b)" # Le motif RE minimum
REGEX_ENTETE = r"(?:^|\n\s{0,4}|^\s{1,4}|\n\|\s{1,4})"
MOTIF_RE_SOUPLE_BUT_AT_THE_BEGINNING_OF_SEQUENCES = REGEX_ENTETE+r"(\bécart\b|\becart\b|\bremarque\b|\be\b|\br(?!(\.|\s+\d+-\d+))\b)"
MOTIF_RE_W_NUMBER = r"\b(ecart|écart|remarque|e|r)(?:\s+){0,2}(?:\s+\w+){0,2}(?:\s+){0,2}(?:\s+(n|numéro|numero))?(?:\s+){0,2}(?:°)?(?:\s+){0,2}\d{1,2}"
MOTIF_RE_W_NUMBER_AND_COLON = r"\b(ecart|écart|remarque|e|r)(?:\s+){0,2}(?:\s+\w+){0,2}(?:\s+){0,2}(?:\s+(n|numéro|numero))?(?:\s+){0,2}(?:°)?(?:\s+){0,2}\d{1,2}\s+.{0,20}(:|\|)" # Je rajoute .{0,20} pour les cas où y'a genre Ecart n 25 Les Jardins :
MOTIF_RE_W_COLON_AND_DOT =  r"\b(ecart|écart|remarque|e|r)(?:\s+){0,2}(?:\s+\w+){0,2}(?:\s+){0,2}(?:\s+(n|numéro|numero))?(?:\s+){0,2}(?:°)?(?:\s+){0,2}\d{0,2}\s+.{0,20}(:|\|)" # Je rajoute .{0,20} pour les cas où y'a genre Ecart n 25 Les Jardins :

sentences_unflag = ['Ce champ de contrôle compte 7 écarts et 6 remarques.', 'LISTE RECAPITULATIVE DES ECARTS (Ecart) ET DES REMARQUES (Remarque)\xa0:',\
     'Les écarts et les remarques formulés dans le présent rapport sont fondés sur les constats et analyses réalisés par la mission sur les champs du contrôle et les appréciations portées sont formulées par la mission en niveau S.A.M.I*.',
     'A = Acceptable, maîtrise correcte du risque maltraitance, mais existence d’écarts à la norme ou de défauts mineurs dans son identification, sa formalisation, sa mise en œuvre et son évaluation sans conséquence directe toutefois sur la santé, la sécurité et l’accompagnement des usagers.',
 'PARTIE I\xa0: Observations motivant des écarts ou des remarques',
  '► les remarques sont l’expression écrite d’un défaut plus ou moins grave perçu par le(s)inspecteur(s) et qui ne peut être caractérisé par rapport à un référentiel législatif ou réglementaire mais en référence notamment aux recommandations de bonnes pratiques professionnelles validées. Ces dernières sont l’expression des pratiques attendues des professionnels dans l’exercice de leur métier en l’état des connaissances scientifiques et des principes éthiques actuels ainsi que de la déontologie propre à leur métier.',
 'Ce document expose les remarques et écarts observés par les inspecteurs :',
 'Remarque (mentionnée ci-après en italique) = observation qui ne peut pas être caractérisée par un référentiel opposable.',
 'Ecart (mentionné ci-après en gras) = non-conformité constatée par rapport aux référentiels opposables.',
 'Les écarts sont ensuite classés en critique, majeur ou mineur :',
 'Ecart critique : présente un danger grave pour la santé ou les droits des résidents. Une mesure corrective immédiate est nécessaire;',
 'Ecart majeur : déviation importante au regard des référentiels opposables, pourrait présenter des dangers pour la pour la santé ou les droits des résidents. Une mesure corrective rapide est nécessaire ;',
 'Ecart mineur : écart par rapport à un référentiel opposable. Une mesure corrective est nécessaire.',
 "REMARQUES",
]