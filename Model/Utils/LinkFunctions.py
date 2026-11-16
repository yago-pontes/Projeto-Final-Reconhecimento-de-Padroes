import numpy as np
from abc import ABC, abstractmethod
from typing import Any, override

class LinkFunction(ABC):
    @abstractmethod
    def Link(self, mu: np.ndarray)->np.ndarray:
        pass
    @abstractmethod
    def Inverse(self, mu: np.ndarray)->np.ndarray:
        pass
    @abstractmethod
    def Derivative(self, mu: np.ndarray)->np.ndarray:
        pass
    @abstractmethod
    def Deviance(self, y_pred: np.ndarray, y_true: np.ndarray)->np.ndarray:
        pass
    @staticmethod
    def IterativelyReweightedLeastSquares(X: np.ndarray, W: np.ndarray, Z: np.ndarray)->np.ndarray:
        XtW = X.T @ W
        # Solve (X^T W X) * beta = (X^T W Z) directly instead of inverting
        return np.linalg.solve(XtW @ X, XtW @ Z)

class Logit(LinkFunction):
    @override
    def Link(self,mu:np.ndarray)->np.ndarray:
        """
        Função de ligação da Binomial/Bernoulli, ligação Logit.
        """
        mu = np.clip(mu, 1e-10, 1-1e-10)
        return np.log(mu / (1.0-mu))
    @override
    def Inverse(self, mu: np.ndarray)->np.ndarray:
        """
        Inversa da função de ligação da binomial/bernoulli, ligação logit inversa.
        """
        return 1/(1+np.exp(-mu))
    @override
    def Derivative(self, mu: np.ndarray)->np.ndarray:
        """
        Derivada da função de ligação da binomial/bernoulli, derivada ligação logit.
        """
        mu = np.clip(mu, 1e-10, 1-1e10)
        return 1/(mu*(1-mu))
    @override
    def Deviance(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calcula a deviance para o modelo Logit.
        y_pred: array de previsões do modelo.
        y_true: array de valores reais.
        Returns:
            A deviance do modelo Logit (float).
        """

        epsilon = 1e-15
        cdf_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        deviance = -2 * np.sum(y_true * np.log(cdf_pred) + (1 - y_true) * np.log(1 - cdf_pred))
        
        return deviance

class Inverse(LinkFunction):
    @override    
    def Link(self, mu: np.ndarray)->np.ndarray:
        """
        Função de ligação da Gamma, ligação inversa.
        """
        mu = np.clip(mu, 1e-10, None)
        return 1/mu
    @override
    def Inverse(self, mu: np.ndarray)->np.ndarray:
        """
        Inversa da função de ligação da Gamma, ligação inversa inversa.
        """
        mu = np.clip(mu, 1e-10, None)
        return 1/mu
    @override
    def Derivative(self, mu: np.ndarray)->np.ndarray:
        """
        Derivada da função de ligação da Gamma, derivada ligação inversa.
        """
        mu = np.clip(mu, 1e-10, None)
        return -1/mu**2
    @override
    def Deviance(self, y_pred: np.ndarray, y_true: np.ndarray)->np.floating:
        """
        Calcula a deviance para o modelo Inverso.
        y_pred: array de previsões do modelo.
        y_true: array de valores reais.
        Returns:
            A deviance do modelo Inverso (float).
        """
        if np.any(y_pred <= 0):
            raise ValueError("As previsões devem ser estritamente positivas para o modelo Inverso.")

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, None)
        
        Deviance = 2 * np.sum( -np.log(y_true / y_pred) + (y_true - y_pred) / y_pred)

        return Deviance

class Identity(LinkFunction):
    @override
    def Link(self, mu: np.ndarray)->np.ndarray:
        return mu
    @override
    def Inverse(self, mu: np.ndarray)->np.ndarray:
        return mu
    @override
    def Derivative(self, mu: np.ndarray)->np.ndarray:
        return np.ones(len(mu))
    @override
    def Deviance(self, y_pred: np.ndarray, y_true: np.ndarray)->np.floating:
        """
        Calcula o erro quadrático médio entre as previsões e os valores reais, que na prática é a deviance escalonada.
        y_pred: array de previsões do modelo.
        y_true: array de valores reais.
        Returns:
            O erro quadrático médio (float).
        """
        return np.sum((y_pred - y_true) ** 2)

class Log(LinkFunction):
    @override
    def Link(self, mu: np.ndarray)->np.ndarray:
        """
        Função de ligação da Poisson, ligação logarítmica.
        """
        mu = np.clip(mu, 1e-10, 1-1e-10)
        return np.log(mu)
    @override
    def Inverse(self, mu: np.ndarray)->np.ndarray:
        """
        Inversa da função de ligação da Poisson, ligação logarítmica inversa.
        """
        return np.exp(mu)
    @override
    def Derivative(self, mu: np.ndarray)->np.ndarray:
        """
        Derivada da função de ligação da Poisson, derivada ligação logarítmica.
        """
        mu = np.clip(mu, 1e-10, 1-1e-10)
        return 1/mu
    @override
    def Deviance(self, y_pred: np.ndarray, y_true: np.ndarray)->np.floating:
        """
        Calcula a deviance para o modelo Logarítmico.
        y_pred: array de previsões do modelo.
        y_true: array de valores reais.
        Returns:
            A deviance do modelo Logarítmico (float).
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, None)

        terms = np.zeros_like(y_true, dtype=float)
        mask = y_true > 0
        terms[mask] = (
            y_true[mask] * np.log(y_true[mask] / y_pred[mask])
        )

        Deviance = 2*np.sum(terms - (y_true - y_pred))

        return Deviance