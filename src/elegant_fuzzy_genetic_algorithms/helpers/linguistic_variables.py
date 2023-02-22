import simpful as sf

def init_best_fitness():
    Tra_1 = sf.TrapezoidFuzzySet(a=-1, b=0, c=0, d=0.25, term="excellent")
    Tri_1 = sf.TriangleFuzzySet(a=0, b=0.25, c=0.5, term="acceptable")
    Tra_2 = sf.TrapezoidFuzzySet(a=.25, b=.5, c=1, d=2, term="poor")
    best_fitness = sf.LinguisticVariable([Tra_1, Tri_1, Tra_2], universe_of_discourse=[0, .5])
    return best_fitness


def init_avg_fitness():
    Tra_1 = sf.TrapezoidFuzzySet(a=-1, b=0, c=0, d=0.5, term="excellent")
    Tri_1 = sf.TriangleFuzzySet(a=0, b=0.5, c=1, term="acceptable")
    Tra_2 = sf.TrapezoidFuzzySet(a=.5, b=1, c=1, d=2, term="poor")
    avg_fitness = sf.LinguisticVariable([Tra_1, Tri_1, Tra_2], universe_of_discourse=[0, 1])
    return avg_fitness

def init_avg_fit_change():
    Tra_1 = sf.TrapezoidFuzzySet(a=-1, b=0, c=0, d=0.2, term="low")
    Tri_1 = sf.TriangleFuzzySet(a=0, b=0.3, c=.6, term="medium")
    Tri_2 = sf.TriangleFuzzySet(a=0.3, b=1, c=1.6, term="high")


    avg_fitness_change = sf.LinguisticVariable([Tra_1, Tri_1, Tri_2], universe_of_discourse=[0, 1])
    return avg_fitness_change


def init_x_rate():
    Tra_1 = sf.TrapezoidFuzzySet(a=-1, b=0, c=0, d=0.5, term="low")
    Tri_1 = sf.TriangleFuzzySet(a=0, b=0.5, c=1, term="medium")
    Tra_2 = sf.TrapezoidFuzzySet(a=.5, b=1, c=1, d=2, term="high")
    x_rate = sf.LinguisticVariable([Tra_1, Tri_1, Tra_2], universe_of_discourse=[0, 1])
    return x_rate

def init_m_rate():
    Tra_1 = sf.TrapezoidFuzzySet(a=-1, b=0, c=0, d=0.025, term="low")
    Tri_1 = sf.TriangleFuzzySet(a=0, b=0.025, c=.05, term="medium")
    Tra_2 = sf.TrapezoidFuzzySet(a=.025, b=.05, c=.05, d=.5, term="high")
    m_rate = sf.LinguisticVariable([Tra_1, Tri_1, Tra_2], universe_of_discourse=[0, .05])
    return m_rate

def init_subpop_size():
    Tra_1 = sf.TrapezoidFuzzySet(a=-1, b=.14, c=.14, d=0.2, term="small")
    Tri_1 = sf.TriangleFuzzySet(a=.14, b=0.2, c=.26, term="medium")
    Tra_2 = sf.TrapezoidFuzzySet(a=.2, b=.26, c=.26, d=.5, term="large")
    subpop_size = sf.LinguisticVariable([Tra_1, Tri_1, Tra_2], universe_of_discourse=[.14, .26])
    return subpop_size

def init_cf():
    Tra_1 = sf.TrapezoidFuzzySet(a=-1, b=0, c=0, d=0.5, term="excellent")
    Tri_1 = sf.TriangleFuzzySet(a=0, b=0.5, c=1, term="acceptable")
    Tra_2 = sf.TrapezoidFuzzySet(a=.5, b=1, c=1, d=2, term="poor")
    cf = sf.LinguisticVariable([Tra_1, Tri_1, Tra_2], universe_of_discourse=[0, 1])
    return cf


def init_priority():
    Tri_1 = sf.TriangleFuzzySet(a=-2, b=-1, c=-.5, term="veryLow")
    Tri_2 = sf.TriangleFuzzySet(a=-1, b=-.5, c=0, term="Low")
    Tri_3 = sf.TriangleFuzzySet(a=-.5, b=0, c=.5, term="Medium")
    Tri_4 = sf.TriangleFuzzySet(a=0, b=.5, c=1, term="High")
    Tri_5 = sf.TriangleFuzzySet(a=.5, b=1, c=2, term="veryHigh")

    priority = sf.LinguisticVariable([Tri_1, Tri_2, Tri_3, Tri_4, Tri_5], universe_of_discourse=[-1, 1])
    return priority
