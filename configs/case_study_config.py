import numpy as np

# Case Study configuration settings
train_N_iter = [25, 50, 100, 250, 500, 1000, 2000]
k_iter = [1/12, 2, 3, 4, 5, 8, 10, 12]

# Ishigami function
ishigami_settings = {
    "CS_type": "UQ",
    "Name": "Ishigami",
    "rep_iter": train_N_iter,
    "ModelOpts": {
        'Type': 'Model',
        'ModelFun': 'casestudies.models.ishigami.model',
        'Parameters': {
            'a': 7,
            'b': 0.1
            },
        'isVectorized': True
    },
    "InputOpts": {
        'Marginals': [{
            'Name': f'x_{i}',
            'Type': 'Uniform',
            'Parameters': [-np.pi, np.pi]
        } for i in range(1, 4)]
    },
    "N": 500,
    "seed": 1337
}

# Sobol' G function
sobol_settings = {
    "CS_type": "UQ",
    "Name": "Sobol",
    "rep_iter": train_N_iter,
    "ModelOpts": {
        'Type': 'Model',
        'ModelFun': 'casestudies.models.sobol-g.model',
        'Parameters': {
            'dim': 8,
            'P': [1, 2, 5, 10, 20, 50, 100, 500]
            },
        'isVectorized': True
    },
    "InputOpts": {
        'Marginals': [{
            'Name': f'x_{i}',
            'Type': 'Uniform',
            'Parameters': [0., 1.]
        } for i in range(1, 9)]
    },
    "N": 500,
    "seed": 1337
}

# Borehole
borehole_settings = {
    "CS_type": "UQ",
    "Name": "Borehole",
    "rep_iter": train_N_iter,
    "ModelOpts": {
        'Type': 'Model',
        'ModelFun': 'casestudies.models.borehole.model',
        'isVectorized': True
    },
    "InputOpts": {
        'Marginals': [
            {
                'Name': 'rw',  # Radius of the borehole
                'Type': 'Gaussian',
                'Parameters': [0.10, 0.0161812]  # (m)
            },
            {
                'Name': 'r',  # Radius of influence
                'Type': 'Lognormal',
                'Parameters': [7.71, 1.0056]  # (m)
            },
            {
                'Name': 'Tu',  # Transmissivity, upper aquifer
                'Type': 'Uniform',
                'Parameters': [63070, 115600]  # (m^2/yr)
            },
            {
                'Name': 'Hu',  # Potentiometric head, upper aquifer
                'Type': 'Uniform',
                'Parameters': [990, 1110]  # (m)
            },
            {
                'Name': 'Tl',  # Transmissivity, lower aquifer
                'Type': 'Uniform',
                'Parameters': [63.1, 116]  # (m^2/yr)
            },
            {
                'Name': 'Hl',  # Potentiometric head , lower aquifer
                'Type': 'Uniform',
                'Parameters': [700, 820]  # (m)
            },
            {
                'Name': 'L',  # Length of the borehole
                'Type': 'Uniform',
                'Parameters': [1120, 1680]  # (m)
            },
            {
                'Name': 'Kw',  # Borehole hydraulic conductivity
                'Type': 'Uniform',
                'Parameters': [9855, 12045]  # (m/yr)
            },
        ]
    },
    "N": 500,
    "seed": 1337
}

# Airfoil self-noise
airfoil_settings = {
    "CS_type": "ML",
    "Name": "Airfoil",
    "rep_iter": [1/20, 1/10, 1/5, 1/3, 2, 3, 5, 10],
    "file": "/home/moo/Documents/ETH/2023_Herbst/Master_Thesis/Benchmark_Code"
            "/case_studies/models/airfoil_self_noise.dat",
    "typ": "csv",
    "sep": '\t',
    "header": None,
    "names": ["frequency", "attack_angle", "chord_length", "free_stream_vel",
              "suction_side_disp_thick", "scaled_sound_pressure"]
    }


# Abalone
abalone_settings = {
    "CS_type": "ML",
    "Name": "Abalone",
    "rep_iter": k_iter,
    "file": "/home/moo/Documents/ETH/2023_Herbst/Master_Thesis/Benchmark_Code"
            "/case_studies/models/abalone_dataset.data",
    "typ": "csv",
    "sep": ',',
    "header": None,
    "names": ["sex", "length", "diameter", "height", "whole_weight",
              "shucked_weight", "viscera_weight", "shell_weight", "rings"],
    "exclude_cols": ["sex"]
    }

# Ailerons
ailerons_settings = {
    "CS_type": "ML",
    "Name": "Ailerons",
    "rep_iter": [1/20, 1/10, 1/5, 1/3, 2, 3, 5, 10],
    "file": "/home/moo/Documents/ETH/2023_Herbst/Master_Thesis/Benchmark_Code"
            "/case_studies/models/ailerons.arff",
    "typ": "arff",
    "names": ['climbRate', 'Sgz', 'p', 'q', 'curPitch', 'curRoll', 'absRoll',
              'diffClb', 'diffRollRate', 'diffDiffClb', 'SeTime1', 'SeTime2',
              'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7',
              'SeTime8', 'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12',
              'SeTime13', 'SeTime14', 'diffSeTime1', 'diffSeTime2',
              'diffSeTime3', 'diffSeTime4', 'diffSeTime5', 'diffSeTime6',
              'diffSeTime7', 'diffSeTime8', 'diffSeTime9', 'diffSeTime10',
              'diffSeTime11', 'diffSeTime12', 'diffSeTime13', 'diffSeTime14',
              'alpha', 'Se', 'goal']
}

# Concrete
concrete_settings = {
    "CS_type": "ML",
    "Name": "Concrete",
    "rep_iter": [1/10, 1/5, 1/3, 2, 3, 5, 10],
    "file": "/home/moo/Documents/ETH/2023_Herbst/Master_Thesis/Benchmark_Code"
            "/case_studies/models/Concrete_Data.xls",
    "typ": "xlsx",
    "header": 1,
    "names": ['cement', 'slag', 'ash', 'water', 'superplasticizer',
              'coarse_aggregate', 'fine_aggregate', 'age', 'strength']
}

# CPU_activity
cpu_activity_settings = {
    "CS_type": "ML",
    "Name": "CPU_activity",
    "rep_iter": [1/20, 1/10, 1/5, 1/3, 2, 3, 5, 10],
    "file": "/home/moo/Documents/ETH/2023_Herbst/Master_Thesis/Benchmark_Code"
            "/case_studies/models/dataset_2183_cpu_act.arff",
    "typ": "arff",
    "names": ['lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec',
              'rchar', 'wchar', 'pgout', 'ppgout', 'pgfree', 'pgscan', 'atch',
              'pgin', 'ppgin', 'pflt', 'vflt', 'runqsz', 'freemem', 'freeswap',
              'usr', ]
}

# Wine_Quality
wine_quality_settings = {
    "CS_type": "ML",
    "Name": "Wine_Quality",
    "rep_iter": [1/20, 1/10, 1/5, 1/3, 2, 3, 5, 10],
    "file": "/home/moo/Documents/ETH/2023_Herbst/Master_Thesis/Benchmark_Code"
            "/case_studies/models/wine_quality.arff",
    "typ": "arff",
    "names": ['fixed_acidity', 'volatile_acidity', 'citric_acid',
              'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
              'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol',
              'quality']
}

case_study_config = {
    "ishigami": ishigami_settings,
    "sobol": sobol_settings,
    "borehole": borehole_settings,
    "airfoil": airfoil_settings,
    "abalone": abalone_settings,
    "ailerons": ailerons_settings,
    "concrete": concrete_settings,
    "cpu_activity": cpu_activity_settings,
    "wine_quality": wine_quality_settings
}
