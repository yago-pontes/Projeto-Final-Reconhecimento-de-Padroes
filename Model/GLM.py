import numpy as np
from LinkFunctions import *
from Aderencia import *
class MLGBase:
    """
    Classe do GLM base sem predição conforme.

    Parâmetros:
    ---
    fit_intercept: bool : Intercepto b no preditor linear
    verbose: bool : Verbosidade do modelo
    max_iter: int : número máximo de iterações do modelo
    tol: float : Tolerância para a mudança de valor na função de perda escolhida
    dist: str : Distribuição da variável resposta, distribuições possíveis->
    
    "norm": distribuição normal

    "ber": distribuição bernoulli

    "poiss": distribuição poisson

    "bin": distribuição binomial

    "gamma": distribuição gamma

    Métodos:
    ---
    
    fit(X: np.ndarray, y:np.ndarray)->None
    predict(X: np.ndarray)->np.ndarray
    
    """

    def __init__(self, dist: str | None = None, fit_intercept: bool = True, verbose: bool = False, max_iter: int = 200, tol: float = 1e-10):
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.dist = None if dist == None else dist
        self.beta = None
        self.ligacao = None
        self.aderencia: dict | None
        self.carleman : bool

    def __IdentificaDist(self, y: np.ndarray)->None:
        if not AdhesionTest.CarlemanCriteria(y):
            self.carleman = False
            print("Condição de carleman não satisfeita, momentos podem não determinar unicamente a distribuição. ")
        else:
            self.carleman = True
        stats = AdhesionTest.ChooseDist(y)
        self.dist = stats["dist"]
        if self.verbose:
            print(f"Distribuição {stats["dist"]} escolhida automaticamente. ")
        if self.dist == "poiss":
            test = Poiss()
        elif self.dist == "ber":
            test = Binom()
        elif self.dist == "gamma":
            test = Gamma()
        elif self.dist == "norm":
            test =  Norm()
        if len(y)>100:
            p = []
            ad = []
            for _ in range(30):
                y_n = np.random.choice(y, size=100)
                dictdados = self.__Aderencia(y_n, test)
                p.append(dictdados["p-valor"])
                ad.append(dictdados["estatistica"])
            self.aderencia = {
                "estatistica": np.mean(ad),
                "p-valor": np.mean(p)
            }
        else:
            self.aderencia = self.__Aderencia(y, test)
    
    def __Aderencia(self, y:np.ndarray, test: AdhesionTest)->dict:
        return test.Adhesion(y)
    
    def fit(self, X: np.ndarray, y:np.ndarray)->"MLGBase":
        if not self.dist:
            self.__IdentificaDist(y)
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        _, features = X.shape
        if self.dist == "poiss":
            self.ligacao = Log(self.dist)
        elif self.dist == "ber":
            self.ligacao = Logit(self.dist)
        elif self.dist == "gamma":
            self.ligacao = Inverse(self.dist)
        elif self.dist == "norm":
            self.ligacao =  Identity(self.dist)
        else:
            raise ValueError("Distribuição desconhecida")
        eta = self.ligacao.Link(self.__mu_inicial(y, self.dist))
        self.beta = np.linalg.lstsq(X, eta, rcond=None)[0]
        return self.__fit(X, y, self.ligacao)
    
    def __fit(self, X:np.ndarray, y: np.ndarray, ligacao: LinkFunction)->"MLGBase":
        for i in range(self.max_iter):
            eta = X @ self.beta
            mu = ligacao.Inverse(eta)
            w = 1/(ligacao.Variance(mu)*(ligacao.Derivative(mu)**2))
            W = np.diag(w)
            z = eta + (y-mu)*ligacao.Derivative(mu)
            self.beta = LinkFunction.IterativelyReweightedLeastSquares(X,W,z)
            previous = ligacao.Deviance(mu, y)
            nmu = ligacao.Inverse(X @ self.beta)
            current = ligacao.Deviance(nmu, y)
            if np.abs(previous - current) < self.tol:
                if self.verbose:
                    print(f"Modelo convergiu na iteração {i} com Deviance {current:.4f}")
                return self
            previous = current
        if self.verbose:
            print(f"Modelo convergiu na iteração {i} com Deviance {current:.4f}")
        return self
    
    def predict(self, X: np.ndarray)->np.ndarray:
        if self.beta is None or self.ligacao is None:
            raise Exception("Você não treinou o modelo!")
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        eta = X @ self.beta
        return self.ligacao.Inverse(eta)

    def __mu_inicial(self, y: np.ndarray, dist: str) -> np.ndarray:
        if dist == "poiss":
            return y + 0.1
        if dist == "ber":
            return (y + 0.5) / 2
        if dist == "gamma":
            return np.maximum(y, 1e-3)
        return y.astype(float)