import argparse
import logging
import json
import os
import utils
from pathlib import Path


def main():
    # Retrieving arguments
    parser = argparse.ArgumentParser(description='Continous integration in Saagie Project')
    parser.add_argument("--action", type=str, choices=['package_job', 'update_job', 'update_pipeline', 'run_job'],
                        help="Action to do with job: 'package_job', 'update_job', 'update_pipeline'",
                        required=True)
    parser.add_argument("--job_name", type=str,
                        help="Name of the job", required=False)
    parser.add_argument("--pipeline_name", type=str,
                        help="Name of the pipeline", required=False)
    parser.add_argument("--saagie_url", type=str,
                        help="URL of Saagie Platform", required=False)
    parser.add_argument("--saagie_user", type=str,
                        help="Saagie user", required=False)
    parser.add_argument("--saagie_pwd", type=str,
                        help="Saagie user's password", required=False)
    parser.add_argument("--saagie_realm", type=str,
                        help="Saagie_realm", required=False)
    parser.add_argument("--saagie_env", type=str,
                        help="Saagie environment", required=False, default="dev")
    parser.add_argument("--job_config_folder", type=str,
                        help="Folder where job config files are stored",  default="./saagie/jobs/*.json")
    parser.add_argument("--pipeline_config_folder", type=str,
                        help="Folder where pipeline config files are stored", default="./saagie/pipelines/*.json")
    parser.add_argument("--env_config_folder", type=str,
                        help="Folder where env config files are stored", default="./saagie/envs/*.json")
    parser.add_argument("--job_source_folder", type=str,
                        help="Folder where job source files are stored", default="./code/jobs/*/*")
    parser.add_argument("--artefact_code_folder", type=str,
                        help="Folder where artefact code files are stored", default="./dist/*/*")
    parser.add_argument("--debug", help="Enable debug mode", action="store_const",
                        dest="loglevel", const=logging.DEBUG, default=logging.INFO)

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel, format="%(asctime)s - [%(levelname)s] - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True)

    if not args.job_name and not args.pipeline_name:
        logging.warning("You must specify a job or a pipeline name")
        exit(0)

    # Retrieving job config
    if "job" in args.action:
        with open(Path(args.job_config_folder).parents[0] / f"{args.job_name}.json", "r") as f:
            job_config = json.load(f)

    if args.action == "package_job":
        if job_config["file_path"]:
            utils.package_code(Path(args.artefact_code_folder).parents[1] / args.job_name / args.job_name, Path(args.job_source_folder).parents[1] / args.job_name)
            logging.info(f"Successfully package job: [{args.job_name}]")
        else:
            logging.info(f"There is no corresponding artefact path for the job: [{args.job_name}]")
        return

    # Retrieving environment config
    with open(Path(args.env_config_folder).parents[0] / f"{args.saagie_env}.json", "r") as f:
        env_config = json.load(f)

    if args.action == "update_job":
        client_saagie = utils.connect_to_saagie(args.saagie_url,
                                                env_config["platform_id"],
                                                args.saagie_user,
                                                args.saagie_pwd,
                                                args.saagie_realm)
        utils.create_or_upgrade_job(client_saagie,
                                    Path(args.job_config_folder).parents[0] / f"{args.job_name}.json",
                                    Path(args.env_config_folder).parents[0] / f"{args.saagie_env}.json")

    if args.action == "run_job":
        client_saagie = utils.connect_to_saagie(args.saagie_url,
                                                env_config["platform_id"],
                                                args.saagie_user,
                                                args.saagie_pwd,
                                                args.saagie_realm)
        utils.run_job(client_saagie,
                      Path(args.job_config_folder).parents[0] / f"{args.job_name}.json",
                      Path(args.env_config_folder).parents[0] / f"{args.saagie_env}.json")

    if args.action == "update_pipeline":
        client_saagie = utils.connect_to_saagie(args.saagie_url,
                                                env_config["platform_id"],
                                                args.saagie_user,
                                                args.saagie_pwd,
                                                args.saagie_realm)
        utils.create_or_upgrade_graph_pipeline(client_saagie,
                                               Path(args.pipeline_config_folder).parents[0] / f"{args.pipeline_name}.json",
                                               Path(args.env_config_folder).parents[0] / f"{args.saagie_env}.json")

    if args.action == "run_pipeline":
        client_saagie = utils.connect_to_saagie(args.saagie_url,
                                                env_config["platform_id"],
                                                args.saagie_user,
                                                args.saagie_pwd,
                                                args.saagie_realm)
        utils.run_pipeline(client_saagie,
                           Path(args.pipeline_config_folder).parents[0] / f"{args.pipeline_name}.json",
                           Path(args.env_config_folder).parents[0] / f"{args.saagie_env}.json")
    logging.info("DONE")


if __name__ == "__main__":
    main()
