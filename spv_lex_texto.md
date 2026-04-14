

# Sistema de Processamento Visual - Laboratório Experimental (SPVLex)



## Integrantes do grupo:



- 11201811864 - Lucas Araujo Avelino Dos Santos

- 11201920812 - Vithório da Cunha Marques

- 11202231732 - Wagner Ryu Kamiya



#### Data de elaboração deste laboratório: 10/04/2026 a 14/04/2026



#### Pré-requisitos: Jupyter Notebook, Python 3.x, NumPy, OpenCV, Matplotlib, image_transformers.py (biblioteca autoral baseada em NumPy e OpenCV)

---


# Instruções de setup do ambiente



# Pipeline de Processamento de Imagens com OpenCV



Este projeto implementa um sistema de **processamento de imagens** utilizando OpenCV, com suporte a execução em imagens estáticas e em tempo real (webcam), além de visualização passo a passo dos resultados. Para organizar e reaproveitar o código, extraímos as funções principais nos scripts render_pipeline.py (com código de renderização das imagens) e image_transformes.py (que contém as funções de transformação)



---



# criar ambiente

python -m venv venv



# ativar (Linux/Windows)

source venv/bin/activate  # ou venv\Scripts\activate



# instalar dependências

pip install opencv-python numpy matplotlib notebook ipykernel



# executar

jupyter notebook



# executar o código

Execute as células na ordem em que aparecem.

---


# Introdução



Neste roteiro você irá executar as transformações utilizadas para o desenvolvimento do SPV através da webcam e interação com a interface gráfica.

Explore as transformações disponíveis, observe e descreva os resultados.



Ao executar o código, serão exibidas duas janelas:

1. Janela com todas as transformações executadas.

2. Janela com os sliders para as transformações. Os valores dos sliders são exibidos na imagem acima. Aumente o tamanho da janela para melhorar a visibilidade conforme necessário.



Altere os valores dos sliders e observe as mudanças nas imagens.

---


## Resumo simplificado das transformações utilizadas na pipeline



| Etapa           | Função                           |
|-----------------|----------------------------------|
| Blur            | Remove ruído                     |
| Sobel           | Detecta bords                   |
| Otsu            | Binariza                         |
| GenerateMask    | Gera múltiplas máscaras          |
| Erode           | Limpa máscaras                   |
| HLS             | Trabalha com cor                 |
| ApplyHueWithMask| Aplica cores nas regiões         |
| RGB             | Volta ao formato final           |

---


## Resumo simplificado do efeito dos sliders

| Slider         | O que muda na imagem        |
| -------------- | --------------------------- |
| gaussian_size  | suavidade geral             |
| gaussian_sigma | intensidade do blur         |
| sobel_size     | espessura das bordas        |
| sobel_scale    | força das bordas            |
| morph_size     | limpeza / remoção de ruído  |
| masks          | tamanho/alcance das regiões |
| hue            | cor aplicada                |
| mode           | comportamento global        |


---


# Questionário

[link](https://docs.google.com/forms/d/e/1FAIpQLSdgvNzoRHQRrUCcegpVv2dcc71vW9NpHBQhLVzH-1S1kNxegQ/viewform?usp=publish-editor)



# Link da enquete de opinião

[link](https://docs.google.com/forms/d/e/1FAIpQLSdmMp4R27tze1bmE_mCLsnK7CHCH_Nyyg3CB2ixoGt7IdYL4A/viewform?usp=publish-editor)

---

