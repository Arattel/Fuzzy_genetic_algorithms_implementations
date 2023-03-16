import fuzzylite as fl


age = fl.InputVariable(
    'age',
    enabled=True,
    minimum=0.0,
    maximum=1.0, 
    terms=[
            fl.Triangle("teenager", 0.25, 0.45, 0.65),
            fl.Triangle("adult", 0.45, 0.65, 0.85),
            fl.Trapezoid("infant", -1, 0, .25, 0.45),
            fl.Trapezoid("elderly", .65, .85, 1, 2),
        ],
)

diversity = fl.InputVariable(
    'diversity', 
    minimum=0, 
    maximum=10, 
    terms=[
        fl.Triangle("medium", 2.5, 4.5, 6.5),
        fl.Triangle('low',4.5, 6.5, 8.5), 
        fl.Trapezoid('high', -1, 0, 2.5, 4.5),
        fl.Trapezoid('very_low', 6.5, 8.5, 10, 20)
    ]
    )

partner_age = fl.OutputVariable(
    'partner_age',
    enabled=True,
    minimum=0.0,
    maximum=1.0, 
    terms=[
            fl.Triangle("teenager", 0.25, 0.45, 0.65),
            fl.Triangle("adult", 0.45, 0.65, 0.85),
            fl.Trapezoid("infant", -1, 0, .25, 0.45),
            fl.Trapezoid("elderly", .65, .85, 1, 2),
        ],
    defuzzifier=fl.Centroid(),
    aggregation=fl.Maximum(),
)