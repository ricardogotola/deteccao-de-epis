# Detecção de Equipamentos de Proteção Individual (EPIs)
<h1>MVP da Pós graduação - Ciências de Dados e Analytics</h1>

![image](https://github.com/user-attachments/assets/bf569c88-ba7d-4a7a-b274-dad50b19cb6b)

# Segurança no Trabalho e Gestão de EPIs

## Introdução
  A segurança no trabalho é um dos pilares fundamentais para o sucesso de grandes obras, sendo diretamente responsável pela proteção dos trabalhadores e pela redução de acidentes e lesões. Nesse contexto, os Equipamentos de Proteção Individual (EPIs) desempenham um papel essencial, pois são projetados para proteger o trabalhador contra riscos específicos presentes no ambiente de trabalho.
  
  No entanto, em obras de grande porte, onde a diversidade de atividades e a complexidade dos riscos são intensificadas, a utilização de EPIs distintos em áreas diferentes pode apresentar desafios significativos para a segurança e a saúde dos trabalhadores
  
  A necessidade de adaptar os EPIs às condições específicas de cada área de trabalho – como setores de soldagem, eletricidade, construção pesada ou movimentação de cargas – pode gerar situações de incompatibilidade, desconforto, e até mesmo falhas na proteção oferecida. Além disso, a troca constante de equipamentos e a gestão logística dessas proteções aumentam a complexidade no controle e monitoramento de sua utilização.
  
  Em face desses desafios, é crucial compreender como a aplicação e a gestão dos EPIs em grandes obras podem afetar a eficácia da segurança no trabalho, a produtividade dos operários e, principalmente, a prevenção de acidentes.
  
  Este trabalho sobre machine learning visa explorar os problemas decorrentes do uso de diferentes tipos de EPIs em áreas diversas de grandes obras, discutindo os impactos na segurança dos trabalhadores e propondo soluções para uma gestão mais eficiente e integrada desses equipamentos. A partir de uma análise detalhada dos desafios enfrentados no cotidiano das obras, serão apresentados caminhos para aprimorar a proteção no ambiente de trabalho, assegurando a integridade física dos trabalhadores e contribuindo para um ambiente de trabalho mais seguro, eficiente e inteligente.

## Objetivo
<p style="text-align: justify;">O objetivo deste projeto é desenvolver um sistema automatizado, por meio de um totem equipado com tecnologia de visão computacional, capaz de verificar se o colaborador está utilizando todos os Equipamentos de Proteção Individual (EPIs) necessários para a área de trabalho específica em que se encontra.</p>

<p style="text-align: justify;">O totem será capaz de realizar a identificação visual do trabalhador e, com base nos critérios predefinidos para cada setor da obra, realizar a inspeção e confirmação do uso correto dos EPIs. Este sistema atuará como uma ferramenta complementar aos profissionais de segurança do trabalho, proporcionando uma verificação rápida, precisa e constante, minimizando os riscos de falhas humanas e aumentando a eficiência na garantia da segurança.</p>

<p style="text-align: justify;">Além disso, o totem ajudará a reforçar a cultura de segurança, oferecendo uma solução prática e ágil para o controle e monitoramento do uso adequado dos equipamentos de proteção nas diversas áreas da obra.</p>

## Descrição do Problema
<p style="text-align: justify;">A utilização de EPIs diferentes em áreas distintas de grandes obras pode gerar desafios na proteção dos trabalhadores, aumentando o risco de falhas no uso adequado dos equipamentos e complicando a gestão da segurança, especialmente em ambientes com múltiplos riscos.</p>

## Tipo de Problema

<p style="text-align: justify;">Este é um problema de classificação supervisionada, onde o modelo aprenderá a associar para solucionar o problema. Será utilizada uma Inteligência Artificial baseada em visão computacional, capaz de identificar e verificar automaticamente se o trabalhador está utilizando todos os EPIs necessários para a área de trabalho específica.</p>

### • Premissa:
<p style="text-align: justify;">Em grandes obras, a diversidade de riscos e a utilização de diferentes Equipamentos de Proteção Individual (EPIs) em áreas distintas tornam desafiadora a verificação constante e eficiente do uso adequado desses equipamentos pelos trabalhadores, o que pode comprometer a segurança no ambiente de trabalho.</p>

### • Hipótese:
<p style="text-align: justify;">Se for implementado um sistema baseado em visão computacional, como um totem automatizado, para verificar a conformidade no uso dos EPIs, será possível reduzir significativamente o risco de falhas na proteção dos trabalhadores, ao proporcionar uma verificação rápida, precisa e contínua, auxiliando os profissionais de segurança do trabalho no controle e monitoramento dos equipamentos necessários para cada área.</p>

**Desafios:** 
**1)** <p style="text-align: justify;">O maior desafio enfrentado durante o trabalho foi a obtenção de um _dataset_ com imagens adequadas para o treinamento do modelo, além da necessidade de aprender e dominar novas ferramentas para realizar o aprendizado de máquina com imagens. Isso envolveu não apenas a rotulagem precisa dos dados, mas também a tarefa de separar as imagens de boa qualidade das de qualidade inferior, o que exigiu um processo cuidadoso de curadoria e preparação do conjunto de dados.</p>
**2)** <p style="text-align: justify;">Outro fator desafiador foi o tempo necessário para treinar o modelo. Em diversos momentos, parecia que o processo havia travado ou falhado, já que não havia nenhuma indicação de progresso. Mesmo assim, confiei e esperei pacientemente cerca de 54 horas para a conclusão do treinamento.</p>

### Elaboração do _dataset_:
Foi utilizada uma ferramenta em _Python_ chamada `bing-image-downloader` para expandir o _dataset_, com o objetivo de aumentar a quantidade de imagens e, assim, melhorar a precisão na detecção dos EPIs nos colaboradores. A ferramenta permite a coleta rápida de imagens diretamente da web, contribuindo para a diversidade e qualidade do conjunto de dados. Para utilizá-la, basta executar o seguinte comando no _Python_ para realizar a instalação:

```bash
pip install bing-image-downloader
```

## Instalação do LabelImg

Para instalar o `LabelImg`, é necessário ter o _Python_ e o `pip` instalados no sistema. Em seguida, execute o comando abaixo no terminal:

```bash
pip install labelImg
```

## Pesquisa sobre os principais algoritmos para detecção de objetos em tempo real

### Principais algoritmos para detecção de objetos em tempo real

Para este trabalho, foram pesquisados os três principais algoritmos de detecção de objetos em tempo real: _YOLO_ (_You Only Look Once_), _SSD_ (_Single Shot MultiBox Detector_) e _RetinaNet_. Abaixo, segue uma explicação breve de cada um:

1. **YOLO (_You Only Look Once_):**
   - Detecção feita em uma única etapa, processando a imagem como um todo.
   - Divide a imagem em uma grade e prevê diretamente as _bounding boxes_ e classes.
   - É extremamente rápido e eficiente, ideal para aplicações em tempo real.

2. **SSD (_Single Shot MultiBox Detector_):**
   - Usa uma arquitetura multiescala, detectando objetos em diferentes níveis de detalhe.
   - Gera caixas ancoradas predefinidas, ajustando-as durante o treinamento.
   - Equilibra velocidade e precisão, mas pode ser menos eficiente que o _YOLO_ para objetos pequenos.

3. **RetinaNet:**
   - Combina uma _Feature Pyramid Network_ (FPN) com ancoragem multiescala.
   - Usa a _Focal Loss_ para lidar com classes desbalanceadas, focando em exemplos mais difíceis.
   - É mais preciso para objetos pequenos e raros, mas menos veloz que os outros dois.

## Tabela Comparativa

| Característica                     | YOLO                                   | SSD                                       | RetinaNet                                |
|-------------------------------------|----------------------------------------|-------------------------------------------|------------------------------------------|
| **Velocidade**                      | Muito rápida (ótima para tempo real). | Rápida, mas geralmente mais lenta que YOLO.| Relativamente mais lenta devido à Focal Loss. |
| **Precisão em Objetos Pequenos**    | Pode perder alguns objetos pequenos.  | Melhor que YOLO, mas depende da configuração. | Excelente, especialmente em cenários desbalanceados. |
| **Escalabilidade**                  | Fácil de implementar e escalar.       | Mais complexo devido às âncoras.          | Moderado, com maior custo computacional. |
| **Robustez em Classes Raras**      | Média.                                | Média.                                    | Alta (graças à Focal Loss).              |
| **Facilidade de Treinamento**      | Simples (requer poucos ajustes).      | Pode exigir mais ajuste nas âncoras.      | Treinamento mais desafiador.             |

Para este trabalho, selecionamos a rede _YOLO_ devido à sua velocidade, pois ela detecta objetos em tempo real com alta eficiência. Além disso, sua arquitetura simplificada realiza a detecção em uma única etapa, reduzindo significativamente a complexidade computacional. 

É relativamente fácil de configurar e treinar, sendo altamente adaptável para diferentes aplicações, como vigilância, drones e veículos autônomos. Apesar de não ser tão precisa para objetos pequenos quanto outros modelos, apresenta boa precisão geral e grande suporte da comunidade, com recursos disponíveis para implementação em dispositivos limitados.

---

## Preparação do Ambiente para o treinamento e utilização do detector de EPIs

1. **Instalar Pré-requisitos**
   - Instale o _Python_ (3.7 ou superior) e adicione-o ao _PATH_.
   - Instale o _Visual Studio Build Tools_ com suporte a C++.
   - Baixe e instale o _CUDA Toolkit_ (se usar GPU).

## 2) Baixar o Darknet

1. **Abra o terminal do VSCode e clone o repositório:**
   ```bash
   git clone https://github.com/AlexeyAB/darknet.git
   cd darknet
   ```

2.	**Edite o arquivo Makefile:**
-	Defina GPU=1 e CUDNN=1 se usar GPU.
-	Use um editor como o próprio VSCode:
```bash
Makefile
```

3.	**Compile o Darknet:**
```bash
make
```

## 3) Configurar Arquivos YOLO
•	Baixe:
-	yolov3.cfg e coco.names do site oficial ou do repositório Darknet;
-	yolov3.weights;
-	Coloque esses arquivos na pasta darknet;

## 4) Testar o YOLO
•		Executar a detecção em uma imagem no terminal:
```bash
darknet.exe detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
```
•		Visualizar os resultados salvos como predictions.jpg na pasta darknet.

## 5. Usar o YOLO com Python
# 1.	Instale bibliotecas no terminal:
```bash
pip install numpy opencv-python
```

# 2.	Use este exemplo básico para detecção:
```bash
import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

img = cv2.imread("data/dog.jpg")
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
detections = net.forward()

for detection in detections:
    print(detection)  # Exemplo de saída
```

Para este trabalho, utilizamos este passo a passo básico para a instalação do **YOLOv3/Darknet** no **Windows 11** com o **VSCode**. Inicialmente, configuramos o ambiente seguindo as etapas de download do **Darknet**, ajustes no arquivo **Makefile** e instalação dos arquivos necessários (**yolov3.cfg**, **yolov3.weights** e **coco.names**).

Também testamos o funcionamento do modelo realizando a detecção em uma imagem estática, garantindo que tudo estava configurado corretamente.

Após isso, avançamos no desenvolvimento do projeto, adaptando o código para realizar detecção em tempo real usando a câmera, ajustando parâmetros como **confiança mínima** e otimizando o pipeline para capturar e processar frames diretamente do dispositivo de vídeo.

## Execução do projeto

Após finalizarmos todas as instalações e configurações necessárias para o ambiente, iniciamos as etapas para a criação e execução do projeto que será dividida nas seguintes etapas:

1) **Preparar o dataset**;
   - a) Baixar as imagens;
   - b) Fazer a higienização das imagens, removendo imagens que não fazem sentido para o projeto ou são de baixa qualidade;
   - c) Criar o arquivo de texto (“txt”) com os **labels** que utilizaremos;
   - d) Rotular as imagens, criando os **bounding boxes** necessários;
   - e) Separar o dataset em **treino** e **teste**;
   
2) **Treinar o modelo**;
   - a) Preparar o arquivo “**data**” do **YOLO**;
   - b) Configurar o arquivo **config** com a extensão “**cfg**” do **YOLO**;
   - c) Baixar o arquivo de peso padrão do **YOLO**, para este projeto utilizamos o padrão **yolov4.conv.137**;
   - d) Treinar o modelo;
   
3) **Teste do modelo treinado**
   - a) .
   - b) .

4) **Execução do programa com OpenCV2 e modelo YOLO treinado**

5) **Resultados**

6) **Resumo**

### 1) Preparação do Dataset
- a. Foram utilizadas 300 imagens baixadas com o **bing-image-downloader** sendo:
   - i. 100 de **óculos de proteção**;
   - ii. 100 de **capacetes de proteção**;
   - iii. 100 de **abafador de ruídos**;
   
- b. Após a higienização dos dados, removendo imagens com baixa resolução e com informações fora do escopo, o total de imagens ficou conforme abaixo:
   - i. 83 de **óculos de proteção**;
   - ii. 65 de **capacetes de proteção**;
   - iii. 48 de **abafador de ruídos**;
   
- c. Após a higienização as imagens e objetos foram rotulados de acordo com sua finalidade utilizando o **LabelImg**.

- d. Os **datasets** foram divididos na proporção de 70% para **treino** e 30% para **testes**, ficando da seguinte forma:
   - i. **Óculos de proteção**: 58 para **treino** e 25 para **teste**;
   - ii. **Capacetes de proteção**: 45 para **treino** e 20 para **teste**;
   - iii. **Abafador de ruídos**: 38 para **treino** e 10 para **teste**;

### 2) Treinar o Modelo
Com os **datasets** separados, juntei as imagens de **treino** em um único dataset e as imagens de **teste** em outro dataset. O processo de **treino** levou aproximadamente 54 horas para finalizar, gerando o arquivo de modelo **modelo epis_last.weights**, posteriormente utilizarei para a **detecção em tempo real** com a **webcam**.

### 3) Testando o Modelo
O teste foi realizado obtendo os seguintes resultados:

![image](https://github.com/user-attachments/assets/001105f2-4474-41b5-b527-372dff34df04)

No teste realizado acima enviei imagens apenas de abafadores de ruído e Óculos de Proteção.

### 4) Aplicando o modelo com OpenCV

Para este **MVP** utilizei a biblioteca **cv2**, que é o módulo principal do **OpenCV** (**Open Source Computer Vision Library**) em **Python**, uma poderosa biblioteca voltada para **processamento de imagens** e **visão computacional**. O **OpenCV** é amplamente utilizado em projetos que envolvem **análise** e **manipulação de imagens**, **vídeos** e até mesmo em aplicações de **inteligência artificial** e **machine learning**, especialmente para **reconhecimento de objetos**, **rostos** e outros padrões.

A configuração para a utilização do padrão treinado foi a seguinte:

# Caminhos para os arquivos
```bash
weights_path = r"E:\deteccao_EPI\.venv\backup\epis_last.weights"
config_path = r"E:\deteccao_EPI\.venv\epis.cfg"
classes_path = r"E:\deteccao_EPI\.venv\labels.txt"
data_path = r"E:\deteccao_EPI\.venv\epis.data"
```

Desta forma estou utilizando o modelo que treinei com os labels e configurações para a detecção dos objetos treinados no MVP, o código completo ficou conforme abaixo:
```bash
import cv2
import numpy as np

# Caminhos para os arquivos
weights_path = r"E:\deteccao_EPI\.venv\backup\epis_last.weights"
config_path = r"E:\deteccao_EPI\.venv\epis.cfg"
classes_path = r"E:\deteccao_EPI\.venv\labels.txt"
data_path = r"E:\deteccao_EPI\.venv\epis.data"

# Carregar as classes
with open(classes_path, "r") as f:
    classes = f.read().strip().split("\n")

# Carregar o modelo YOLO
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Inicializar a webcam
cap = cv2.VideoCapture(0)  # 0 para a webcam padrão
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Largura do frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Altura do frame

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    (H, W) = frame.shape[:2]

    # Criar um blob do frame (resolução reduzida para melhorar desempenho)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)

    # Obter as saídas da rede
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(output_layers)

    # Analisar as detecções
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Ajuste do limite de confiança
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Max Suppression para eliminar duplicatas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar o frame com as detecções
    cv2.imshow("Detecção de Objetos", frame)

    # Parar o loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
```

### 5)	Conclusão
Apesar de o dataset utilizado ser relativamente pequeno, o tempo de treinamento das imagens foi consideravelmente longo. No entanto, os resultados obtidos foram, no mínimo, promissores. A detecção de óculos de segurança se mostrou a mais eficaz durante a identificação em tempo real pela câmera, enquanto o capacete e o abafador de ruídos apresentaram maior demora para serem reconhecidos.
Com a ampliação do dataset, uma rotulação mais detalhada e um maior refinamento nos critérios de classificação, é possível alcançar melhorias significativas nos resultados. 
Abaixo, apresento algumas imagens capturadas durante os testes de detecção realizados utilizando minha webcam.
