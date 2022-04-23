# Libs - Funções usadas na App Para Fusão de Imagens Médicas de Exames MRI e PET

# Imports
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from typing import List
from cv2.ximgproc import guidedFilter
from scipy.spatial.distance import cdist


# Função para extrair os detalhes da imagem aplicando filtro 
# O processo de filtragem é muito usado para extrair os ruídos, e nesse caso é exatamete o que queremos separar
def filtra_imagem(img, k = (30, 30)):
    
    # Verifica o tipo da imagem (array)
    assert img.dtype == np.float32 and img.max() <= 1., 'imagem precisa ser np.float32'
    
    # Cria o objeto de remoção de ruído com blur
    base = cv2.blur(img, k)
    
    # Faz a operação extraindo o ruído da imagem
    detail = img - base
    
    return base, detail


# Função para decomposição das imagens
def decompose_imagem(imagens_norm):
    
    # Listas para base e detalhes
    bases, details = [], []
    
    # Loop pelas imagens normalizadas
    for img_n in imagens_norm:
        
        # Aplicação a função de filtro
        base, detail = filtra_imagem(img_n)
        
        # Cria a lista de bases das imagens
        bases.append(base)
        
        # Cria a lista de ruídos (detalhes) das imagens
        details.append(detail)
        
    return bases, details


# Função para extrair as saliências das imagens
def extrai_saliencia(img, D):
    
    # Verifica o tipo da imagem
    assert img.dtype == np.uint8, 'imagem precisa estar como np.uint8'
    
    # Conta o número de ocorrências de cada pixel
    # https://numpy.org/doc/stable/reference/generated/numpy.bincount.html
    hist = np.bincount(img.flatten(), minlength = 256) / img.size
    
    # Calcula os valores que representam saliência
    sal_values = np.dot(hist, D)
    
    # Extrai as saliências
    saliency = sal_values[img]
    
    return saliency


# Função que gera os pesos das saliências
def gera_pesos_saliencias(imgs):
    
    # O objeto D será usado para calcular as distâncias euclidianas
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    D = cdist(np.arange(256).reshape(-1, 1), np.arange(256).reshape(-1, 1))
    
    # Os pesos das saliências serão calculdos com base nas saliências e nas distâncias
    Ws = [extrai_saliencia(img, D) for img in imgs]
    
    # Inclui um valor de correção (algo parecido com a taxa de aprendizado)
    # https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
    Ws = np.dstack(Ws) + 1e-12
    
    # Gera os pesos finais
    Ws = Ws / Ws.sum(axis = 2, keepdims = True)
    return Ws


# Função para otimização dos pesos das imagens
# https://en.wikipedia.org/wiki/Edge-preserving_smoothing
# https://github.com/opencv/opencv_contrib/tree/master/modules/ximgproc
def guided_optimize(guides, srcs, r, eps):
    
    # Filtro da imagem mantendo as bordas
    Ws = [guidedFilter(guide.astype(np.float32), src.astype(np.float32), r, eps) for guide, src in zip(guides, srcs)]
    
    # Matriz de peso com fator de correção
    Ws = np.dstack(Ws) + 1e-12
    
    # Matriz final
    Ws = Ws / Ws.sum(axis = 2, keepdims = True)
    
    return Ws


# Função para a soma da multiplicação de pesos pelas bases das imagens
def pesos_imagens(imgs, ws):
    return np.sum(ws * np.dstack(imgs), axis = 2)


# Função que aplica o modelo pré-treinado para encontrar os pesos para fusão dos detalhes das imagens.
def fusao_detalhes_cnn(inp, modelo, device, relus):
    
    # Coloca o modelo no device
    modelo.to(device)
    modelo.eval()

    # Prepara input e output
    inp = inp.to(device)
    out = inp
    
    # Lista de pesos
    Wls = []  
    
    # Loop pelos pesos
    with torch.no_grad():
        
        # Loop pelas relus
        for i in range(max(relus) + 1):
            
            # Extrai as features usando o modelo pré-treinado
            out = modelo.features[i](out)
            
            # Se a camada relu estiver na lista de relus, aplicamos interpolação
            # https://pytorch.org/docs/stable/nn.functional.html
            if i in relus:
                
                # Aplicamos interpolação às imagens para reduzir a escala
                l1_feat = (F.interpolate(out, inp.shape[-2:]).norm(1, dim = 1, keepdim = True))  
                
                # Gera os pesos da camada softmax
                w_l = F.softmax(l1_feat, dim = 0)
                
                # Gera lista de pesos
                Wls.append(w_l)

    # Tensor para receber a saliência máxima
    saliency_max = -np.inf * torch.ones((3,) + inp.shape[-2:])
    saliency_max = saliency_max.to(device)
    
    # Loop pelos pesos para extrair a saliência máxima
    for w_l in Wls:
        saliency_curr = (inp * w_l).sum(0)
        saliency_max = torch.max(saliency_max, saliency_curr)

    # Obtendo a saliência máxima na imagem
    fused_detail = saliency_max
    
    # Retornamos o resultado como array NumPy
    return to_numpy(fused_detail[0])


# Função para converter a imagem de um espaço de cores para outro
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html 
def split_YCbCr(imgs):
    Y, Y_f, CbCr_f = [None] * len(imgs), [None] * len(imgs), [None] * len(imgs)
    for i, img in enumerate(imgs):
        if is_gray(img):
            Y[i] = img
        else:
            YCrCb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            Y[i] = YCrCb[:, :, 0]
            CbCr_f[i] = (YCrCb[:, :, 1:] / 255.).astype(np.float32)
        Y_f[i] = (Y[i] / 255.).astype(np.float32)
    return Y, Y_f, CbCr_f


# Converte a imagem para o espaço de cores original
def YCbCr_to_RGB(CbCrs, fusedY):
    fused = fusedY
    for cbcr in CbCrs:
        if cbcr is not None:
            fused = np.dstack((fusedY, cbcr))
            fused = cv2.cvtColor(fused, cv2.COLOR_YCrCb2RGB)
            fused = np.clip(fused, 0, 1)
    return fused


# Função para converter para formato Numpy
def to_numpy(t):
    a = t.squeeze().detach().cpu().numpy()
    if a.ndim == 3:
        np.moveaxis(a, 0, -1)
    return a


# Função para converter os detalhes das imagens em tensores
def convert_to_tensor(imgs: List[np.ndarray]):
    
    # Lista temporária
    tmp = []
    
    # Loop pelas imagens
    for img in imgs:
        if img.dtype == np.uint8:
            img = (img / 255.).astype(np.float32)
        if img.ndim == 2:
            img = np.expand_dims(img, (0, 1))
            tmp.append(np.repeat(img, 3, axis = 1))
            
    return torch.from_numpy(np.vstack(tmp))


# Leitura da imagem
def read_image(image_path):
    img = cv2.imread(str(image_path), -1)
    if img is None:
        raise FileNotFoundError(f'Erro ao carregar a imagem {str(image_path)}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# Salva a imagem
def save_image(img: np.ndarray, savePath):
    if img.dtype == np.float32:
        assert img.max() <= 1 and img.min() >= 0, f'Imagem de tipo np.float32 deve ter range 0-1'
        img = np.rint(img * 255).astype(np.uint8)
    Image.fromarray(img).save(savePath)


# Verifica a escala da imagem
def is_gray(img: np.ndarray):
    assert len(img.shape) <= 3, 'O shape da imagem está errado'
    if img.ndim == 2 or img.shape[2] == 1:
        return True
    return False


# Função para o stack das imagens
def _c3(img):
    if img.ndim == 2:
        img = np.dstack((img, img, img))
    return img


# Grava texto na imagem
def putText(img, text):
    pos = (15, 25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    img = cv2.putText(img, text, pos, font, 0.8, color, 2)
    return img


# Tamanho do grid
IMSIZE = (360, 270)


# Gera cada linha do grid
def grid_row(*imgs, resized = IMSIZE):
    row = []
    for img in imgs:
        row.append(cv2.resize(img, resized, interpolation = cv2.INTER_CUBIC))
    return row


# Cria o grid final
def make_grid(nested_list, resized = IMSIZE, addText = False, hsep = 7, wsep = 7):
    m, n = len(nested_list), len(nested_list[0])
    w, h = resized

    gH = m * h + (m - 1) * hsep
    gW = n * w + (n - 1) * wsep

    grid = (np.ones((gH, gW, 3)) * 255).astype(np.uint8)
    for i in range(m):
        y = (h + hsep) * i
        for j in range(n):
            x = (w + wsep) * j
            this = _c3(nested_list[i][j])
            if addText:
                text = f'Input-{j+1}' if j < n - 1 else 'Fused'
                this = putText(this, text)
            grid[y:y + h, x:x + w] = this

    return grid
