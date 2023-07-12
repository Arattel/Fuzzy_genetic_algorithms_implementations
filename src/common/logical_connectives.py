import fuzzylite as fl


connectives = {
    'min': fl.Minimum(), 
    'product': fl.AlgebraicProduct(), 
    'max': fl.Maximum(), 
    'sum':fl.AlgebraicSum()
}