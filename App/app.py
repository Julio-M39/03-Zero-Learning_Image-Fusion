# App Para Fusão de Imagens Médicas de Exames MRI e PET

# Imports
import sys
import torch
import argparse
from lib import *
from pathlib import Path
from torchvision.models.vgg import vgg19

# Função para o parse dos argumentos de entrada
def parse_args():
    parser = argparse.ArgumentParser(description = 'App Para Fusão de Imagens Médicas de Exames MRI e PET')
    parser.add_argument('--CaminhoImagens', required = True)
    parser.add_argument('--Imagens', required = True, nargs = '+')
    args = parser.parse_args()
    return args

# Classe para os argumentos usados no processo
class Args:

    # Device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Tamanho do kernel
    k = (30, 30)  

    # Parâmetros do Guided Filter
    r, eps = 45, 0.01  

    # Carrega o modelo pré-treinado
    modelo = vgg19(pretrained = True)

    # Lista de camadas relu
    relus = [1, 3, 8] 

    # Cria o objeto com processamento dos argumentos
    args = parse_args()

    # Caminho para as imagens
    imagePath = Path(args.CaminhoImagens)

    # Lista para as imagens
    imageSources = []

    # Looop pelas imagens passadas no argumento
    for pattern in args.Imagens:
        imageSources.append(sorted(imagePath.glob(pattern)))

    # Cria o bundle de imagens
    bundles = zip(*imageSources)

    # Tamanho do grid para gerar as imagens de saída
    grid_cell_size = (320, 320) 

    # Pasta para gravar o resultado (se executar pela linha de comando)
    # resultPath = Path('../resultado').joinpath(imagePath.stem)

     # Pasta para gravar o resultado (se executar pelo Jupyter Notebook)
    resultPath = Path('resultado').joinpath(imagePath.stem)


# Execução principal do programa
if __name__ == '__main__':

    # Lista para os resultados
    nested_list = []

    # Processa a pasta para gravar os resultados
    Args.resultPath.mkdir(parents = True, exist_ok = True)

    # Loop pelos pares de imagens
    for bundle in Args.bundles:
        print(f'\nFusão entre as imagens => f{[fp.name for fp in bundle]}')

        # Leitura da imagem
        imgs = [read_image(fp) for fp in bundle]  

        # Converte a imagem para outro espaço de cores (específico para trabalhar como imagens coloridas)
        Ys, Ys_f, CbCrs_f = split_YCbCr(imgs)

        # Decomposição da imagem
        bases, details = decompose_imagem(Ys_f)

        # Pesos das saliências
        Wb_0 = gera_pesos_saliencias(Ys)

        # Ajusta o shape
        Wb_0 = np.moveaxis(Wb_0, -1, 0) 

        # Encontra os melhores valores de peso com base no Guided Filter
        Wb = guided_optimize(Ys_f, Wb_0, Args.r, Args.eps)

        # Fusão das bases
        fused_base = pesos_imagens(bases, Wb)

        # Converte os detalhes para tensor
        tensor_details = convert_to_tensor(details)

        # Fusão dos detalhes
        fused_details = fusao_detalhes_cnn(tensor_details, Args.modelo, Args.device, relus = Args.relus)

        # Fusão das bases e dos detalhes
        fusedY_f = np.clip(fused_base + fused_details, 0, 1)

        # Converte a imagem para o espaço de cores original
        fused_f = YCbCr_to_RGB(CbCrs_f, fusedY_f)

        # Converte o resultado para os valores de pixels para gerar a fusão final
        fused_u8 = np.rint(fused_f * 255).astype(np.uint8)

        # Obtém o número da imagem para concatenar ao nome
        name = ''.join(x for x in bundle[0].name if x.isdigit())

        # Salva a imagem em disco
        save_image(fused_u8, Args.resultPath.joinpath(f'FUSED-{name}.png'))

        # Gera o grid das imagens
        nested_list.append(grid_row(*imgs, fused_u8, resized = Args.grid_cell_size))

    # Cria oo grid de imagens
    grid = make_grid(nested_list, Args.grid_cell_size, addText = True)

    # Salva o grid em formato pdf
    save_image(grid, Args.resultPath.joinpath('resultado_combinado.pdf'))

    # Concluído
    print("\nFusão Concluída!\n")


