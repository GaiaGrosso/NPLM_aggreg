import os, json,argparse, datetime, time, glob
import numpy as np

OUTPUT_DIRECTORY = '/n/home00/ggrosso/MMD/out/'

def create_config_file(config_table, OUTPUT_DIRECTORY):
    with open('%s/config.json'%(OUTPUT_DIRECTORY), 'w') as outfile:
        json.dump(config_table, outfile, indent=4)
    return '%s/config.json'%(OUTPUT_DIRECTORY)

# configuration dictionary
config_json = {
    "N_Ref"   : 20000,
    "N_Bkg"   : 2000-110,#-10,#-80,
    "N_Sig"   : 110,#10,#80,
    "Sig_loc" : 6.4,
    "Sig_std" : 0.16,
    "is_NP2": True,
    "is_Pois": False,
    "N_flat_extra" : 0,
    "output_directory": OUTPUT_DIRECTORY,
}
is_pois = ''
if not config_json["is_Pois"]:
    is_pois = 'NoPois_'

if config_json["N_Sig"]:
    if config_json["is_NP2"]:
        ID = 'MMDfuse%iD_%sNR%i_NB%i_NS%i_tail-excess'%(1+config_json["N_flat_extra"], is_pois, config_json['N_Ref'], config_json['N_Bkg'], config_json['N_Sig'])
    else:
        ID = 'MMDfuse%iD_%sNR%i_NB%i_NS%i_loc%s_std%s'%(1+config_json["N_flat_extra"], is_pois, config_json['N_Ref'], config_json['N_Bkg'], config_json['N_Sig'],
                                           str(config_json['Sig_loc']), str(config_json['Sig_std']))
else:
    ID = 'MMDfuse%iD_%sNR%i_NB%i_NS%i'%(1+config_json["N_flat_extra"], is_pois, config_json['N_Ref'], config_json['N_Bkg'], config_json['N_Sig'])

#### launch python script ###########################
if __name__ == '__main__':
    parser   = argparse.ArgumentParser()
    parser.add_argument('-p','--pyscript', type=str, help="name of python script to execute", required=True)
    parser.add_argument('-l','--local',    type=int, help='if to be run locally',             required=False, default=0)
    parser.add_argument('-t', '--toys',    type=int, help="number of toys per jobs",    required=False, default = 1)
    parser.add_argument('-j', '--jobs',    type=int, help="number of jobs submissions",   required=False, default = 100)
    args     = parser.parse_args()
    ntoys    = args.toys
    njobs    = args.jobs
    pyscript = args.pyscript

    config_json['pyscript'] = pyscript
    
    pyscript_str = pyscript.replace('.py', '')
    pyscript_str = pyscript_str.replace('_', '/')
    config_json["output_directory"] = OUTPUT_DIRECTORY+'/'+pyscript_str+'/'+ID
    if not os.path.exists(config_json["output_directory"]):
        os.makedirs(config_json["output_directory"])
    
    json_path = create_config_file(config_json, config_json["output_directory"])
    if args.local:
        os.system("python %s/%s -j %s -t %i"%(os.getcwd(), pyscript, json_path, ntoys))
    else:
        label = "folder-log-jobs"
        os.system("mkdir %s" %label)
        for i in range(njobs):
            script_sbatch = open("%s/submit_%i.sh" %(label, i) , 'w')
            script_sbatch.write("#!/bin/bash\n")
            script_sbatch.write("#SBATCH -c 1\n")
            script_sbatch.write("#SBATCH --gpus 1\n")
            script_sbatch.write("#SBATCH -t 0-0:20\n")
            script_sbatch.write("#SBATCH -p gpu\n")
            script_sbatch.write("#SBATCH --mem=16000\n")
            script_sbatch.write("#SBATCH -o %s/%s"%(label,pyscript_str)+"_%j.out\n")
            script_sbatch.write("#SBATCH -e %s/%s"%(label,pyscript_str)+"_%j.err\n")
            script_sbatch.write("\n")
#            script_sbatch.write("module load python/3.10.9-fasrc01\n")
#            script_sbatch.write("module load cuda/12.2.0-fasrc01\n")
            script_sbatch.write("\n")
            script_sbatch.write("python %s/%s -j %s -t %i\n"%(os.getcwd(), pyscript, json_path, ntoys))
            script_sbatch.close()
            os.system("chmod a+x %s/submit_%i.sh" %(label, i))
            os.system("sbatch %s/submit_%i.sh"%(label, i))
