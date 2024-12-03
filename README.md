Projeto desenvolvido como projeto final do curso de Inteligencia Artificial, ministrado pelo SENAI-Tagutainga-DF, Professor Rômuo Silvestre (https://github.com/romulosilvestre).
com a ajuda de Andre Vargas (https://github.com/AndreVargas0) e Alana Calado (https://github.com/alanacalado)

# Ciclo Lazer - Previsão de Bicicletas Disponíveis

Este projeto utiliza aprendizado de máquina para prever a quantidade de bicicletas disponíveis em estações de bicicletas compartilhadas. O modelo leva em consideração dados históricos de viagens, características dos usuários e informações das estações para realizar previsões.

## Funcionalidades

- **Visualização interativa das estações no mapa**: Utilizando a biblioteca PyDeck, o projeto exibe um mapa interativo com as localizações das estações e a quantidade de bicicletas disponíveis.
- **Previsão de bicicletas disponíveis**: Com base em informações como a estação escolhida, horário e dia da semana, o modelo K-Nearest Neighbors (KNN) prevê a quantidade de bicicletas disponíveis em uma estação específica.
- **Análise de dados históricos**: O sistema processa dados históricos de viagens para identificar padrões e melhorar a precisão das previsões.

## Tecnologias Utilizadas

- **Python 3.x**
- **Streamlit**: Para construção da interface web interativa.
- **Pandas**: Para manipulação e análise de dados.
- **PyDeck**: Para visualização interativa de mapas.
- **Scikit-learn**: Para treinamento do modelo de aprendizado de máquina (KNN).
- **StandardScaler**: Para normalização dos dados antes de alimentar o modelo.

## Estrutura do Projeto

- `datasets/`: Pasta contendo os arquivos CSV com os dados de viagens e estações.
- `ciclolazer.jpg`: Imagem do logo que será exibida na barra lateral.
- `app.py`: Script principal do projeto com a implementação do Streamlit, processamento de dados e previsões.
- `requirements.txt`: Arquivo com as dependências do projeto.

## Como Rodar o Projeto

1. Clone este repositório:
    ```bash
    git clone https://github.com/seu-usuario/ciclo-lazer.git
    ```

2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3. Coloque os arquivos CSV de dados nas pastas corretas (em `datasets/`):
    - `df_rides.csv` com os dados de viagens.
    - `df_stations.csv` com os dados das estações.

4. Execute o aplicativo Streamlit:
    ```bash
    streamlit run app.py
    ```

5. Acesse o aplicativo no seu navegador em `http://localhost:8501`.

## Como Usar

- Selecione as estações que deseja visualizar no mapa através da barra lateral.
- Escolha a estação para a previsão de bicicletas e ajuste o horário e o dia da semana.
- O modelo irá calcular a quantidade prevista de bicicletas disponíveis e exibir no mapa interativo.

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

## Contribuições

Contribuições são bem-vindas! Se você quiser contribuir com melhorias, correções ou novas funcionalidades, sinta-se à vontade para abrir um "pull request".

## Contato

Se você tiver dúvidas ou quiser entrar em contato, pode me encontrar no [LinkedIn]([https://www.linkedin.com/in/seu-perfil/](https://www.linkedin.com/in/luiz-bernardino-89931a1bb/)) ou [GitHub]([https://github.com/seu-usuario](https://github.com/GLuizHB)).
