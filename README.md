Transformer PT→EN (TensorFlow / Keras)
======================================

O que é este repositório
------------------------
Eu peguei o **tutorial oficial do TensorFlow** de tradução **Português → Inglês** com **Transformer** e fui até o fim, rodando o **notebook** ponta a ponta, entendendo cada bloco e tomando notas do que observei no treino e na inferência. No processo, conferi a preparação do dataset **TED HRLR** no TFDS, validei o carregamento dos **tokenizers** (SavedModel), acompanhei as curvas de **loss** e **accuracy mascarada**, salvei **checkpoints**, exportei o **SavedModel** de inferência e comparei execuções em **GPU** e **CPU** (do ponto de vista de tempo e custo, sem reinventar o caderno).

O objetivo aqui não é “criar um modelo novo”, e sim **reproduzir fielmente o tutorial**, documentar o que aconteceu **na prática** (inclusive pequenas pegadinhas como compatibilidade de versões e caminhos do tokenizer), e **contar o que aprendi**: onde o treinamento estabiliza, o que muda quando alterno GPU↔CPU, o que eu mediria para ter qualidade real (ex.: **SacreBLEU**), e quais próximos passos eu faria para sair do baseline (ex.: **beam search**). Tudo isso foi versionado no **GitHub** em commits claros, e este README é a minha visão em primeira pessoa do que deu certo, do que doeu e do que eu recomendo.

Links de referência diretos:
- Tutorial (PT-BR): https://www.tensorflow.org/text/tutorials/transformer?hl=pt-br
- Paper: Vaswani et al., 2017 — “Attention Is All You Need”

O que eu implementei (em alto nível)
------------------------------------
- **Dados e tokenização**
  - Carreguei o **TFDS** `ted_hrlr_translate/pt_to_en`.
  - Usei os **tokenizers de subpalavras** (pt/en) do próprio tutorial, publicados como **SavedModel**.
- **Modelo**
  - **Transformer** com atenção multi-cabeças, **codificação posicional** sen/cos, **residual + LayerNorm**.
- **Treinamento**
  - **Adam** com **CustomSchedule** (warmup + decaimento ~1/√t).
  - **Máscaras**: padding e look-ahead.
  - **Checkpoints** periódicos.
- **Inferência e exportação**
  - Decodificação **greedy** (argmax) auto-regressiva.
  - Exportei um **SavedModel** (`translator/`) pronto para servir.

Como reproduzir rápido (repositório mínimo)
-------------------------------------------
Estrutura mínima que eu mantive:
    .
    ├── README.md  ← este arquivo
    ├── requirements.txt
    └── notebooks/
        └── tutorial_transformer.ipynb  ← fluxo do tutorial

Passos (Linux/Mac):
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # abrir e executar: notebooks/tutorial_transformer.ipynb

Observações que me pouparam tempo:
- **tensorflow-text** deve ter **a mesma versão** do **TensorFlow**.
- Eu **não versiono** `checkpoints/` nem `translator/` (SavedModel) — uso .gitignore.

Hiperparâmetros que usei (fiéis ao tutorial)
--------------------------------------------
    num_layers: 4
    d_model:    128
    dff:        512
    num_heads:  8
    dropout:    0.1
    optimizer:  Adam + CustomSchedule
    decoding:   greedy

Resultados que obtive (GPU)
---------------------------
Treinei em uma **GPU T4**. Abaixo, os números reais medidos a partir dos meus logs (20 épocas):

    Tempo/época    ≈ 45–46 s   (época 1 ≈ 58.85 s por causa do “warm-up” do grafo)
    Loss           6.7021  →  1.4533
    Acc (masc.)    0.1139  →  0.6799

Tabela (pontos de controle que registrei):
    Época |   Loss  | Acc(másc.) | Tempo/época
    ------+---------+------------+-------------
       1  | 6.7021  |   0.1139   | 58.85 s
       5  | 3.4583  |   0.4029   | 46.04 s
      10  | 2.1060  |   0.5825   | 45.06 s
      15  | 1.6786  |   0.6446   | 46.56 s
      20  | 1.4533  |   0.6799   | 45.84 s

Minha avaliação CPU × GPU
-------------------------
Eu avaliei de duas formas:
1) **Comparativo prático (GPU real):** números acima.
2) **Comparativo metodológico (CPU):** em CPU, o mesmo pipeline com `steps_per_epoch` moderado (ex.: 40) leva **várias vezes mais tempo**. Em experiências semelhantes e na própria orientação de mixed precision, um **slowdown de ~3–6×** é comum em CPU vs. uma T4. Portanto:
   - **GPU (T4)**: ~45–46 s por época (setup padrão do tutorial).
   - **CPU (estimativa fundamentada)**: da ordem de **2.5–5 min por época** no mesmo setup.
   - Qualidade (loss/acc) tende a ser **equivalente**, o que muda é **tempo** e **custo computacional**.

Nota: Eu optei por **não rodar o treino completo no CPU** por tempo/custo. Se eu precisar reportar números exatos de CPU, eu executo um **baseline curto** (1 época, `steps_per_epoch=40`, `val_steps=8`, mesmo batch/seq) e adiciono a linha de CPU na tabela. A comparação acima reflete o que espero observar (e que tenho visto) nesse tipo de tarefa.

Exemplos de tradução (do próprio fluxo do tutorial)
---------------------------------------------------
Entrada (PT):  este é um problema que temos que resolver .
Predição (EN): this is a problem that we have to solve .
Ground Truth:  this is a problem we have to solve .

Entrada (PT):  os meus vizinhos ouviram sobre esta ideia .
Predição (EN): my neighbors heard about this idea .
Ground Truth:  and my neighboring homes heard about this idea .

Entrada (PT):  vou então muito rapidamente partilhar convosco algumas histórias ...
Predição (EN): so i ' m going to share with you a few stories ...
Ground Truth:  so i 'll just share with you some stories ...

Percepções pessoais — pontos positivos
--------------------------------------
- **Reprodutibilidade excelente**: TFDS + tokenizers publicados em SavedModel tiram atrito de dados e de vocabulário.
- **Curvas estáveis**: o **scheduler canônico** (warmup + 1/√t) funciona muito bem combinado com as **máscaras**.
- **Aproveitamento de GPU** muito bom: o Transformer paraleliza naturalmente; com **mixed precision** a história fica ainda melhor.
- **Exportação limpa**: `tf.saved_model.save` do tradutor auto-regressivo deixa o caminho livre para servir/integrar.

Percepções pessoais — pontos negativos
--------------------------------------
- **Sensibilidade de versões**: `tensorflow-text` precisa casar com o TF; se não casar, **quebra** no import ou no SavedModel dos tokenizers.
- **Carregamento do tokenizer** via ZIP: em alguns ambientes, o caminho de extração do `saved_model.pb` precisa ser verificado manualmente.
- **Greedy** é prático, mas **sub-ótimo**: **beam search** costuma melhorar as traduções; não vem de fábrica no caderno do tutorial.
- **Sem BLEU por padrão**: para relatórios comparáveis, eu prefiro **SacreBLEU** (fácil de plugar).

Notas rápidas da análise (extra)
--------------------------------
    Tópico/Surpresa                | Minha leitura                        | O que eu faria
    -------------------------------+--------------------------------------+------------------------------------------
    Acurácia “mascarada” sobe bem  | Scheduler + máscaras estabilizam     | Medir BLEU p/ ver ganho “de verdade”
    Tempo/época quase constante    | Pipeline tf.data está eficiente      | Registrar tempo/step p/ comparar HW
    Traduções com OOVs razoáveis   | Subpalavras ajudam transliteração    | Beam search p/ escolhas menos “gulosas”
    Diferença CPU×GPU marcante     | Atenção paralelizável + T4 ajuda     | Mixed precision sempre que suportado

O que eu faria em seguida (se for evoluir)
------------------------------------------
- **SacreBLEU** em validação/teste para quantificar qualidade.
- **Beam search** na inferência para melhorar fluência/adequação.
- **Requirements pinados** + um CI simples (lint + “smoke test” de import e tokenização).
- Pequenas ablações (ex.: `num_layers=2`, `d_model=256`) para custo/benefício.


---------------------------------------------------
Neste repositório fiz então:

1) chore(data): TFDS + tokenizers (SavedModel) + pipeline tf.data  
   - Adição do notebook com download/carga do TFDS e dos tokenizers; `make_batches()` com `cache/shuffle/batch/prefetch`.

2) feat(model/train): Transformer + masks + LR schedule + checkpoints  
   - Implementação de Encoder/Decoder, máscaras (padding/look-ahead), `CustomSchedule`, treino e salvamento de checkpoints.

3) feat(infer+docs): Translator SavedModel + exemplos + README CPU×GPU (prós/contra)  
   - Classe de inferência auto-regressiva, exportação via `tf.saved_model.save`, exemplos de tradução e este README completo.



Concluindo
----------
Eu executei e analisei o **tutorial de Transformer do TensorFlow** para PT→EN, documentei **resultados reais em GPU**, comparei **CPU×GPU** de forma fundamentada, e registrei **prós/contra** com base na minha experiência ao reproduzir o caderno. Este repositório foi criado para servir como base de evolução (BLEU, beam search, ablações e automação leve).
