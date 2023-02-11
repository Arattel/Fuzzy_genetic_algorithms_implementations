import numpy as np
import simpful as sf




def calculate_lifetime(L, U, fitness, age):
    alpha = (U - L) / 2
    beta = (U + L) / 2
    N_SAMPLES = fitness.shape[0]

    f_avg = fitness.mean()
    f_min = fitness.min()
    f_max = fitness.max()

    phi_small = (fitness - f_min) / (f_avg - f_min)
    phi_big = (fitness - f_avg) / (f_max - f_avg)
    tau = f_avg - fitness

    lifetime = np.zeros(N_SAMPLES)
    lifetime[tau >= 0] = (L + alpha * phi_small[tau >= 0]) 
    lifetime[tau < 0] = (beta + alpha * phi_big[tau < 0])
    return age / lifetime


def init_age():
    Tri_1 = sf.TriangleFuzzySet(a=0.25, b=0.45, c=0.65, term="teenager")
    Tri_2 = sf.TriangleFuzzySet(a=0.45, b=0.65, c=0.85, term="adult")
    Tra_1 = sf.TrapezoidFuzzySet(a=-1, b=0, c=.25, d=0.45, term="infant")
    Tra_2 = sf.TrapezoidFuzzySet(a=.65, b=.85, c=1, d=2, term="elderly")
    age = sf.LinguisticVariable([Tra_1, Tri_1, Tri_2, Tra_2], universe_of_discourse=[0, 1])
    return age


def init_diversity():
    Tri_1 = sf.TriangleFuzzySet(a=2.5, b=4.5, c=6.5, term="medium")
    Tri_2 = sf.TriangleFuzzySet(a=4.5, b=6.5, c=8.5, term="low")
    Tra_1 = sf.TrapezoidFuzzySet(a=-1, b=0, c=2.5, d=4.5, term="high")
    Tra_2 = sf.TrapezoidFuzzySet(a=6.5, b=8.5, c=10, d=20, term="very_low")
    diversity = sf.LinguisticVariable([Tra_1, Tri_1, Tri_2, Tra_2], universe_of_discourse=[0, 10])
    return diversity


def calculate_partner_age(fs, age, diversity):
    fs.set_variable("age", age)
    fs.set_variable("diversity", diversity)
    return fs.Mamdani_inference(["partner_age"])['partner_age']


def init_fuzzy_system(rules_file: str = 'rules.txt'):
    FS = sf.FuzzySystem()
    FS.add_linguistic_variable("age", init_age())
    FS.add_linguistic_variable("diversity", init_diversity())
    FS.add_linguistic_variable("partner_age",init_age())
    FS.add_rules_from_file(rules_file, verbose=False)
    return FS