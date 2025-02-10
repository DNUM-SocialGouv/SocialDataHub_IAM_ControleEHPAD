# SocialDataHub - IAM - SARTRE

`python -m text_extraction.main --path_folder data/input --output_excel data/extraction/extraction`

`python -m text_transformation.main --path_in data/extraction/extraction.pkl --path_out data/transform/transform.pkl`

`python cicd_saagie_tool/__main__.py --action package_job --job_name print_hello_world`

`python cicd_saagie_tool/__main__.py --action update_job --job_name print_hello_world --saagie_url "$SAAGIE_URL" --saagie_user "$SAAGIE_USER" --saagie_pwd "$SAAGIE_PWD" --saagie_realm "$SAAGIE_REALM" --saagie_env dev`

attention ligne 12 de cicd.../main.py : 
`python 
    parser.add_argument("--action", type=str, choices=['package_job', 'update_job', 'update_pipeline', 'run_job'],
`
Il n'y a pas toute les actions donc il faut les ajouter à la liste. 

`python cicd_saagie_tool/__main__.py --action run_job --job_name print_hello_world --saagie_url "$SAAGIE_URL" --saagie_user "$SAAGIE_USER" --saagie_pwd "$SAAGIE_PWD" --saagie_realm "$SAAGIE_REALM" --saagie_env dev`

`python cicd_saagie_tool/__main__.py --action package_job --job_name extract_transform_infer`

`python cicd_saagie_tool/__main__.py --action update_job --job_name extract_transform_infer --saagie_url "$SAAGIE_URL" --saagie_user "$SAAGIE_USER" --saagie_pwd "$SAAGIE_PWD" --saagie_realm "$SAAGIE_REALM" --saagie_env dev`

`python cicd_saagie_tool/__main__.py --action run_job --job_name extract_transform_infer --saagie_url "$SAAGIE_URL" --saagie_user "$SAAGIE_USER" --saagie_pwd "$SAAGIE_PWD" --saagie_realm "$SAAGIE_REALM" --saagie_env dev`