import simpful as sf
import fuzzylite as fl

best_fitness =  fl.InputVariable(
        name="bestFitness",
        description="",
        enabled=True,
        minimum=0.0,
        maximum=.5,
        lock_range=False,
        terms=[
            fl.Trapezoid("excellent", -1, 0, 0, 0.25),
            fl.Triangle("acceptable", 0, 0.25, 0.5,),
            fl.Trapezoid("poor", .25, .5, 1, 2),
        ],)

average_fitness = fl.InputVariable(
        name="avgFitness",
        description="",
        enabled=True,
        minimum=0.0,
        maximum=1.0,
        lock_range=False,
        terms=[
            fl.Trapezoid("excellent", -1, 0, 0, 0.5),
            fl.Triangle("acceptable", 0, 0.5, 1,),
            fl.Trapezoid("poor", .5, 1, 1, 2),
        ],)

average_fitness_change =  fl.InputVariable(
        name="avgFitChange",
        description="",
        enabled=True,
        minimum=0.0,
        maximum=1.0,
        lock_range=False,
        terms=[
            fl.Trapezoid("low", -1, 0,  0, 0.2),
            fl.Triangle("medium", 0, 0.3, .6,),
            fl.Triangle("high", 0.3, 1, 1.6),
        ],)

x_rate = fl.OutputVariable(
        name="xRate",
        description="",
        enabled=True,
        minimum=0.4,
        maximum=1.0,
        lock_range=False,
        defuzzifier=fl.Centroid(),
        aggregation=fl.Maximum(),
        terms=[
            fl.Trapezoid("low", -1, .4, .4, .7,),
            fl.Triangle("medium", .4, 0.7, 1),
            fl.Trapezoid("high", .7, 1, 1, 2),
        ],)


m_rate = fl.OutputVariable(
        name="mRate",
        description="",
        enabled=True,
        minimum=0,
        maximum=.05,
        lock_range=False,
        defuzzifier=fl.Centroid(),
        aggregation=fl.Maximum(),
        terms=[
            fl.Trapezoid("low", -1, 0,  0, 0.025),
            fl.Triangle("medium", 0, 0.025, .05),
            fl.Trapezoid("high", .025, .05, .05, .5),
        ],)


subpop_size = fl.OutputVariable(
        name="subPopSize",
        description="",
        enabled=True,
        minimum=.14,
        maximum=.26,
        lock_range=False,
        defuzzifier=fl.Centroid(),
        aggregation=fl.Maximum(),
        terms=[
            fl.Trapezoid("small", -1, .14, .14, 0.2,),
            fl.Triangle("medium",.14, 0.2, .26,),
            fl.Trapezoid("large", .2,.26,.26, .5),
        ],)

first_cl = fl.InputVariable(
        name="first",
        description="",
        enabled=True,
        minimum=0.0,
        maximum=1.0,
        lock_range=False,
        terms=[
            fl.Trapezoid("excellent", -1, 0, 0, 0.5),
            fl.Triangle("acceptable", 0, 0.5, 1,),
            fl.Trapezoid("poor", .5, 1, 1, 2),
        ],)

second_cl = fl.InputVariable(
        name="second",
        description="",
        enabled=True,
        minimum=0.0,
        maximum=1.0,
        lock_range=False,
        terms=[
            fl.Trapezoid("excellent", -1, 0, 0, 0.5),
            fl.Triangle("acceptable", 0, 0.5, 1,),
            fl.Trapezoid("poor", .5, 1, 1, 2),
        ],)

priority = fl.OutputVariable(
        name="priority",
        description="",
        enabled=True,
        minimum=-1,
        maximum=1,
        lock_range=False,
        defuzzifier=fl.Centroid(),
        aggregation=fl.Maximum(),
        terms=[
            fl.Triangle("veryLow",-2, -1, -.5),
            fl.Triangle("Low",-1, -.5, 0),
            fl.Triangle("Medium",-.5, 0, .5,),
            fl.Triangle("High", 0, .5, 1,),
            fl.Triangle("veryHigh",.5, 1, 2),
        ],)