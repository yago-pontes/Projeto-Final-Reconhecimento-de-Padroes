import numpy as np
from scipy.stats import kurtosis, skew, chi2, binom, goodness_of_fit, chisquare
from abc import ABC, abstractmethod

class AdhesionTest(ABC):

    @abstractmethod
    def Adhesion(self, y: np.ndarray)->dict:
        pass
    @staticmethod
    def CarlemanCriteria(y:np.ndarray, k: int)->bool:
        """
        Avalia se a sequência y satifaz o critério de Carleman para a determinabilidade da distribuição
        de probabilidade a partir de seus k momentos.
    
        Parâmetros:
        y: np.ndarray: Vetor da variável resposta (1D array).
        k: int: Número de momentos a serem considerados.
        
        Retorna:
        bool: True se a sequência satisfaz o critério de Carleman, False caso contrário.
        """
        log_termos = []
        for i in range(1, k + 1):
            m_2i = np.mean(y**(2 * i))
            val = - (1.0 / (2 * i)) * np.log(m_2i + 1e-15)
            log_termos.append(val)
            
        log_termos = np.array(log_termos)
    
        razoes = np.diff(log_termos)
        
        diferencas = np.diff(razoes)
        
        condicao_satisfeita = np.all(diferencas >= -1e-4)
        
        return bool(condicao_satisfeita)
    @staticmethod
    def ChooseDist(y: np.ndarray)->dict:
        """
        Escolhe a distribuição para a variável resposta y de acordo com o suporte
        da variável.
        
        Parâmetros:
        y: np.ndarray: Vetor da variável resposta (1D array).
        
        Retorna:
        dict: Dicionário com estatísticas úteis e distribuição sugerida no final.
        
        Formato do dicionário de saída:
        
        {
        "media":média
        "variancia":variância
        "assimetria":assimetria
        "curtose":curtose
        "dist":distribuição sugerida
        }
        """
        media = np.mean(y)
        variancia = np.var(y,ddof=1)
        assimetria = skew(y)
        curtose = kurtosis(y)
    
        if len(np.unique(y))==2:
            distribuicao = "ber"
        elif np.max(y)<=1 and np.min(y)>=0:
            distribuicao = "bin"
    
        elif media != 0 and np.min(y)>=0:
            if variancia/media < 1.25 and variancia/media > 0.8:
                distribuicao = "poiss"
            else:
                distribuicao = "norm"
    
        elif np.min(y)<=0:
            if np.abs(assimetria)<=0.5:
                distribuicao = "norm"
            else:   
                distribuicao = "norm"
        else:
            distribuicao = "gamma"
    
        dicionario = {
            "media" : media,
            "variancia" : variancia,
            "assimetria" : assimetria,
            "curtose" : curtose,
            "dist" : distribuicao
        }
    
        return dicionario

class Binom(AdhesionTest):
    @override
    def Adhesion(self, y: np.ndarray, n: int)->dict:
        """
        Verifica a aderência da variável resposta a uma distribuição binomial para o parâmetro p.
        Supõe a condição de carleman satisfeita e número de ensaios n conhecido.
        
        Parâmetros:
        y: np.ndarray : Vetor da variável resposta (1D array).
        n: int : número de ensaios de bernoulli da variável resposta.
    
        Retorno:
        lista: {
        "estatistica" : estatística_usada,
        "p-valor" : p-valor
        }
        O valor observado na estatística qui-quadrado usado e o p-valor correspondente.
        """
        tamanho_amostra = len(y)
        p = np.mean(y) / n
        freq_obs = np.bincount(y, minlength=n+1)
    
        k_valores = np.arange(n + 1)
        prob_esperada = binom.pmf(k_valores, n, p)
        freq_esp = tamanho_amostra * prob_esperada
    
        teste1 = chisquare(f_obs=freq_obs, f_exp=freq_esp, ddof=1)
    
        return {
            "estatistica": teste1.statistic,
            "p-valor": teste1.pvalue
        }
class Poiss(AdhesionTest):
    @override
    def Adhesion(self, y: np.ndarray) -> dict:
        """
        Verifica a aderência da variável resposta a uma distribuição poisson para o parâmetro lambda.
        Supõe a condição de carleman satisfeita.
        
        Parâmetros:
        y: np.ndarray : Vetor da variável resposta (1D array).
    
        Retorno:
        lista: {
        "estatistica" : estatística_usada,
        "p-valor" : p-valor
        }
        O valor observado na estatística do teste exato de poisson usado e o p-valor correspondente.
        """
    
        media = np.mean(y)
        var = np.var(y,ddof=1)
        n = len(y)-1
        estatistica = n*var/media
    
        p_valor = 1-chi2.cdf(estatistica, n)
    
        return {
            "estatistica": estatistica,
            "p-valor": p_valor
        }
class Norm(AdhesionTest):
    @override
    def Adhesion(self, y: np.ndarray) -> dict:
        """
        Verifica a aderência da variável resposta a uma distribuição normal para os parâmetros mu e sigma.
        Supõe a condição de carleman satisfeita.
        
        Parâmetros:
        y: np.ndarray : Vetor da variável resposta (1D array).
    
        Retorno:
        {
        "estatistica": anderson-darling,
        "p-valor" : pvalor
        }
        Onde 'estatistica' é a estatística de aderson-darling e p-valor o p-valor observado.
        """
        
        res = goodness_of_fit(dist=norm, data=y, statistic="ad", n_mc_samples=1000)
        
        return {
                "estatistica": res.statistic,
                "p-valor": res.pvalue
            }
class Gamma(AdhesionTest):
    @override
    def Adhesion(self, y: np.ndarray) -> dict:
        """
        Verifica a aderência da variável resposta a uma distribuição gamma para os parâmetros alpha e beta.
        Supõe a condição de Carleman satisfeita.

        Parâmetros:
        y: np.ndarray : Vetor da variável resposta (1D array)

        Retorno:
        {
        "estatistica" : anderson-darling,
        "p-valor" : pvalor
        }
        """
        
        res = goodness_of_fit(dist=gamma, data=y, statistic="ad", n_mc_samples=1000)
        
        return {
                "estatistica": res.statistic,
                "p-valor": res.pvalue
            }
