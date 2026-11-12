# Começar a usar (Windows, sem Docker)

## 1. Instalar uma vez

1. **Python 3.13**, **Node.js 20+** e **ffmpeg**. O ffmpeg pode ser instalado num terminal com:
   ```bash
   winget install Gyan.FFmpeg
   ```
2. Deixe a pasta do projeto **fora do OneDrive** (por exemplo `E:\Dev\ViralCutIA`).
3. Dê dois cliques em `scripts\atualizar-viralcut.bat`. Ele cria o ambiente Python, instala a API,
   instala o site e faz o build. Rode de novo **sempre que atualizar o código** (esta versão
   adiciona a dependência `psutil` e o site precisa ser recompilado).

## 2. Abrir

Dois cliques em `scripts\iniciar-viralcut.bat`. Abrem duas janelas (API e site) e o navegador em
http://localhost:3000. Para fechar, feche as duas janelas.

No **primeiro vídeo**, o modelo de fala é baixado (~1,6 GB) e os de sincronização (~240–380 MB),
para `E:\Dev\ViralCutIA\models`. Durante o download a transcrição fica em 0%: é normal.
O modelo de recorte de pessoa (25 MB) só é baixado se você usar `/tela-verde` (IA) ou
`/texto-atras`.

## 3. Usar

1. **Novo vídeo** → arraste o arquivo.
2. Escolha o **tipo de conteúdo**: Shorts, TikTok, Reels, YouTube longo (16:9) ou Compilação.
3. Três caminhos:
   * **A. Geração rápida**: um clique, a IA escolhe tudo.
   * **B. Presets**: 12 estilos prontos (Podcast, Gaming, Curiosidades, Top 5…).
   * **C. Avançado**: ligue o modo *Advanced* e configure os 13 editores um a um. Música
     (`/trilha`), prints (`/prova`) e vídeo de referência (`/referencia`) são enviados ali mesmo.
4. Acompanhe o progresso; ao terminar, baixe cada MP4/SRT ou tudo em ZIP. O **Growth toolkit**
   de cada clipe traz títulos, descrição, hashtags e capítulos para copiar.
5. **Edit** em um clipe muda cortes, legendas, estilo e editores só daquele clipe; só a renderização
   roda de novo.
6. **Projects** (no topo) lista os vídeos já processados.

## 4. Para não travar o PC

Em *Advanced → Processing mode*:

| Modo | Quando usar |
|---|---|
| **Eco** | Notebook ou usando o PC ao mesmo tempo: metade dos núcleos, prioridade baixa, libera a memória do modelo de fala entre vídeos |
| **Balanced** (padrão recomendado na maioria dos PCs) | Todos os núcleos com prioridade reduzida: seus programas continuam respondendo |
| **Turbo** | PC livre, 4+ núcleos e 8+ GB: renderiza dois clipes ao mesmo tempo |

A qualidade final é a mesma nos três. Os editores marcados *heavy* (`/3d` forte, `/tela-verde`
por IA, `/texto-atras`) aumentam o tempo de render; deixe-os desligados quando não precisar.

## 5. Onde ficam os arquivos

Tudo fica **dentro da pasta do projeto** (`E:\Dev\ViralCutIA`); nada vai para o `C:`:

| O quê | Pasta |
|---|---|
| Vídeos enviados, clipes, banco de dados, cache de transcrições | `apps\api\data` |
| Arquivos temporários (reenquadramento, prévias, ZIPs) | `apps\api\data\tmp` |
| Modelos de IA baixados (~2 GB) | `models` |
| Cache de downloads do pip e do npm | `.cache` |
| Configuração (`.env`) | `apps\api\.env` |

Se mover o projeto para outra pasta ou disco, tudo vai junto: os caminhos são relativos à pasta
do projeto. Depois de mover, rode `scripts\atualizar-viralcut.bat` uma vez (o ambiente Python
guarda o caminho antigo). Para usar outra pasta de modelos, defina `VIRALCUT_MODELS_DIR` no `.env`.

## 6. Bom saber

* Os arquivos são apagados automaticamente após 24 h (`VIRALCUT_RETENTION_HOURS` no `.env`).
* Nada é publicado automaticamente. Use músicas e imagens que você tenha direito de usar.
* Com uma chave da Anthropic (`ANTHROPIC_API_KEY` no `.env`), títulos e avaliação dos trechos
  usam o Claude; sem ela tudo funciona com as heurísticas locais.
* Problemas: a janela "ViralCut API" mostra o log; o botão *Show processing log* no resultado
  mostra cada etapa.
