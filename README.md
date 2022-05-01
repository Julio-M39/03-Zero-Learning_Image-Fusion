# 03-Zero-Learning_Image-Fusion

## **Fusão de Imagens**

A fusão de imagens médicas é o processo de registrar e combinar várias imagens de uma ou várias modalidades de imagem para melhorar a qualidade da imagem e reduzir a aleatoriedade e redundância, a fim de aumentar a aplicabilidade clínica das imagens médicas para diagnóstico e avaliação de problemas médicos.

Em aplicações clínicas, como cirurgia guiada por imagem e diagnóstico não invasivo, dependem fortemente de imagens multimodais. A fusão de imagens médicas desempenha um papel central ao integrar informações de múltiplas fontes em uma única saída mais compreensível. No trabalho <a href="https://arxiv.org/abs/1905.03590/">Fast and Efficient Zero-Learning Image Fusion</a>, os autores propuseram um método de fusão de imagens em tempo real usando redes neurais pré-treinadas para gerar uma única imagem contendo recursos de fontes multimodais. As imagens são mescladas usando uma nova estratégia baseada em mapas profundos extraídos de uma rede neural convolucional.

A imagem abaixo mostra a metodologia proposta pelos autores em seu trabalho.

<div>
<img src="https://user-images.githubusercontent.com/54995990/166152527-d541df63-79ae-4199-8550-0c7bb4882e3b.png" width="980px" />
</div>

### **Resultados**

Após a implementação obtivemos os seguintes resultados que podem ser observados nas imagens abaixo:

Imagens normais
<div>
<img src="https://user-images.githubusercontent.com/54995990/166152887-49dfe82d-56a4-409e-aef3-b4547cfc1f6d.png" width="980px" />
</div>


Imagens médicas
<div>
<img src="https://user-images.githubusercontent.com/54995990/166153064-eb8422f1-bbca-45a7-97cf-137598d063c4.png" width="980px" />
</div>
