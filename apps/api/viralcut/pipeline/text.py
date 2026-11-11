"""Language resources and text utilities shared by the analysis stages.

Lexicons exist for English and Portuguese. For other languages the engine still works from
language-agnostic signals (punctuation, numbers, audio, visuals) and — when configured — the LLM.
All patterns are matched on lower-cased, accent-stripped text, so they are written without accents.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .transcribe import Word

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'’-]*")
_TERMINAL = (".", "?", "!", "…", "。", "？", "！")
_ABBREV = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "sr.",
    "sra.",
    "dra.",
    "prof.",
    "vs.",
    "etc.",
    "st.",
    "jr.",
    "no.",
    "e.g.",
    "i.e.",
}


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def norm(text: str) -> str:
    return strip_accents(text.lower()).replace("’", "'")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(norm(text))


STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        """a about above after again against all also am an and any are aren't as at be because been before being
        below between both but by can can't cannot could did didn't do does doesn't doing don't down during each few
        for from further get gets got had has have having he her here hers herself him himself his how i i'd i'll i'm
        i've if in into is isn't it it's its itself just let's like me more most my myself no nor not of off on once
        only or other our ours out over own really same she should so some such than that that's the their theirs
        them then there there's these they they're this those through to too under until up very was we we're were
        what when where which while who whom why will with would you you'd you'll you're you've your yours yourself
        gonna wanna gotta yeah okay ok oh um uh well thing things going go know think kind sort lot actually
        basically literally right mean one say said says""".split()
    ),
    "pt": frozenset(
        """a o as os um uma uns umas de do da dos das em no na nos nas por para pra pro com sem sob sobre entre ate
        e ou mas que se como quando onde porque pois ja nao sim eu tu ele ela nos voces voce eles elas me te lhe
        meu minha meus minhas seu sua seus suas nosso nossa esse essa esses essas este esta estes estas isso isto
        aquilo aquele aquela la aqui ali ai entao tambem so muito muita muitos muitas mais menos bem mal ser estar
        ter haver foi era sao esta estao estou estava tem tinha tenho vai vou vamos ir fazer faz fez pode ne tipo
        assim cara gente coisa coisas ta to tava tao dai dela dele deles delas num numa pelo pela pelos pelas ao aos
        depois antes ainda agora sempre nunca cada todo toda todos todas outro outra outros outras mesmo mesma
        qual quais quem cujo algo alguem acho sei sabe dizer disse falar falou ah eh hum uh ok""".split()
    ),
}

FILLERS: dict[str, frozenset[str]] = {
    "en": frozenset({"um", "uh", "erm", "hmm", "like", "basically", "literally", "y'know"}),
    "pt": frozenset({"ne", "tipo", "ah", "eh", "hum", "uh", "hm", "assim", "ta"}),
}

# Words that make a sentence depend on what came before it.
DEPENDENT_OPENERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "and",
            "but",
            "because",
            "or",
            "also",
            "which",
            "he",
            "she",
            "they",
            "it",
            "that",
            "those",
            "these",
            "then",
            "though",
            "although",
            "plus",
            "him",
            "her",
            "them",
            "nor",
        }
    ),
    "pt": frozenset(
        {
            "e",
            "mas",
            "porque",
            "ou",
            "tambem",
            "ele",
            "ela",
            "eles",
            "elas",
            "isso",
            "esse",
            "essa",
            "esses",
            "essas",
            "aquilo",
            "dai",
            "porem",
            "pois",
            "alias",
            "inclusive",
            "dele",
            "dela",
        }
    ),
}
MILD_OPENERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {"so", "well", "okay", "ok", "like", "yeah", "right", "now", "anyway", "this", "um", "uh"}
    ),
    "pt": frozenset({"entao", "bom", "tipo", "ai", "ne", "olha", "agora", "enfim", "ah", "eh", "ta"}),
}
# Leading connectives/fillers that can be removed from a quoted sentence without changing its meaning.
STRIPPABLE_OPENERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "and",
            "but",
            "so",
            "because",
            "or",
            "also",
            "then",
            "plus",
            "well",
            "okay",
            "ok",
            "like",
            "yeah",
            "right",
            "now",
            "anyway",
            "um",
            "uh",
        }
    ),
    "pt": frozenset(
        {
            "e",
            "mas",
            "entao",
            "porque",
            "ou",
            "tambem",
            "dai",
            "pois",
            "bom",
            "tipo",
            "ai",
            "ne",
            "olha",
            "agora",
            "enfim",
            "ah",
            "eh",
        }
    ),
}
# Frequent content words that say nothing about the topic (kept out of keywords and hashtags).
GENERIC_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        """every single most many much even still back make made makes making take took takes come came want wanted need
        needed look looked time times way ways people person good great better best thing things something anything
        nothing everything someone anyone everyone first second third next last never always already maybe really
        actually today tomorrow yesterday year years week weeks month months minute minutes hour hours later before
        after only tell told talk talked feel felt give gave keep kept start started work worked little long whole
        different same important sure real true okay right left high going getting doing being having""".split()
    ),
    "pt": frozenset(
        """cada maioria muitos muitas ainda volta fazer feito fazendo tomar tomou vir veio quer quero precisa preciso
        olhar tempo vezes jeito forma maneira pessoa pessoas bom boa melhor coisa coisas algo nada tudo alguem
        ninguem primeiro segundo terceiro proximo ultimo nunca sempre talvez realmente verdade hoje amanha ontem ano
        anos semana semanas mes meses minuto minutos hora horas depois antes apenas falar falou sentir dar deu manter
        comecar comecou trabalhar pouco longo inteiro diferente mesmo importante certo real aqui agora entao porque
        gente parte lado caso exemplo""".split()
    ),
}

TRAILING_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset({"and", "but", "or", "so", "because", "the", "a", "to", "of", "that", "with"}),
    "pt": frozenset({"e", "mas", "ou", "entao", "porque", "o", "a", "de", "que", "com", "para", "pra"}),
}


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9])(?:" + pattern + r")(?![a-z0-9])")


# Hook categories → per-language patterns (accent-less, lower-case).
HOOK_PATTERNS: dict[str, dict[str, re.Pattern[str]]] = {
    "curiosity": {
        "en": _rx(
            r"why|what if|what happens|did you know|have you ever|ever wonder(?:ed)?|the (?:real |actual |main )?(?:reason|truth|secret|problem|trick|difference)"
            r"|(?:nobody|no one) (?:tells|talks|knows|told)|most people (?:don'?t|never|think|fail|get|have no idea)"
            r"|you (?:won'?t|wouldn'?t|will never|would never) (?:believe|guess)|here'?s (?:what|why|the thing)|(?:it )?turns out"
            r"|the (?:one|only|biggest|first|last|single) thing|guess what|secret|mistake|myth|what (?:i|we|they) (?:found|discovered|learned)|how (?:i|we|to|do|does|did|can)"
        ),
        "pt": _rx(
            r"por que|porque sera|e se |o que acontece|voce sabia|sabia que|ja (?:parou|pensou|imaginou|se perguntou)|a (?:grande |real )?verdade"
            r"|o (?:grande |verdadeiro |real )?(?:segredo|motivo|problema|truque)|a diferenca|ninguem (?:te )?(?:conta|fala|sabe|diz|contou)"
            r"|a maioria das pessoas|voce nao vai acreditar|descobri|descobriu|segredo|o (?:maior |pior )?erro|mito|sabe (?:por que|qual|o que|quando|como)"
            r"|como (?:eu|a gente|fazer|voce|ele|ela|funciona|e que)|o que (?:eu|a gente) (?:descobri|aprendi|descobriu|aprendeu)"
        ),
    },
    "interrupt": {
        "en": _rx(
            r"stop|wait|listen|hold on|no way|wrong|lied?|lying|unpopular opinion|hot take|controversial|forget (?:everything|what)"
            r"|never (?:do|say|buy|use)|everyone is wrong|is dead|is a scam|overrated|don'?t (?:do|buy|use|make|ever)|nobody"
        ),
        "pt": _rx(
            r"pare|para de|espera|escuta|calma|errado|errada|mentira|mentiu|opiniao impopular|polemic[oa]|esquece|esqueca"
            r"|nao (?:faca|compre|use|caia)|nunca (?:faca|compre|use|diga)|todo mundo (?:esta|ta) errado|e (?:um )?golpe|superestimad[oa]|ninguem"
        ),
    },
    "stakes": {
        "en": _rx(
            r"lost|lose|losing|risk(?:y|ed)?|danger(?:ous)?|die[ds]?|dying|dead|death|kill(?:ed|s)?|fired|broke|bankrupt|debt|millions?|billions?"
            r"|life[- ]changing|changed my life|worst|disaster|fail(?:ed|ure|s)?|destroy(?:ed)?|crisis"
            r"|emergency|cancer|lawsuit|sued|scam(?:med)?|illegal|arrested|prison|jail|dollars|percent|cost me|ruin(?:ed)?"
        ),
        "pt": _rx(
            r"perdi|perdeu|perder|perdendo|risco|perig(?:o|oso|osa)|morr(?:eu|i|er|endo)|morte|matou|demitid[oa]|quebr(?:ei|ou|ado|ada)"
            r"|fali(?:u|do|r)|divida|dividas|milh(?:ao|oes)|bilh(?:ao|oes)|mil reais|mudou (?:a )?minha vida|pior|desastre"
            r"|fracass(?:o|ei|ou)|larguei|destru(?:iu|ido|ida)|crise|emergencia|cancer|processad[oa]|golpe|ilegal|pres[oa]|cadeia"
            r"|reais|por cento|custou|arruin(?:ou|ei|ado)"
        ),
    },
    "emotion": {
        "en": _rx(
            r"love[ds]?|hate[ds]?|amazing|incredible|insane|crazy|unbelievable|shock(?:ed|ing)?|scared|terrified|afraid|cry(?:ing)?|cried"
            r"|tears|laugh(?:ed|ing)?|hilarious|funny|angry|furious|excited|heartbroken|beautiful|horrible|terrible|awful|wow|oh my god"
            r"|omg|damn|wild|mind[- ]blowing|proud|ashamed|embarrass(?:ed|ing)|nervous|panic(?:ked)?|obsessed|devastat(?:ed|ing)|thrilled"
        ),
        "pt": _rx(
            r"amo|amei|ador(?:o|ei)|odei(?:o|ei)|incrivel|insano|insana|louc(?:o|a|ura)|inacreditavel|choc(?:ado|ada|ante|ou)|medo"
            r"|apavorad[oa]|chor(?:ei|ou|ando|ar)|lagrimas|hilario|engracad[oa]|raiva|furios[oa]|animad[oa]|empolgad[oa]|lind[oa]"
            r"|horrivel|terrivel|meu deus|caramba|caraca|surreal|absurd[oa]|orgulho|orgulhos[oa]|vergonha|nervos[oa]|panico"
            r"|emocion(?:ado|ada|ante|ei)|assustador|assustadora|apaixonad[oa]|desesper(?:o|ado|ada)"
        ),
    },
    "value": {
        "en": _rx(
            r"here'?s how|here (?:is|are)|how to|the (?:best|fastest|easiest|simplest|right) way|tips?|tricks?|steps?|rules?|lessons?"
            r"|strategy|framework|method|formula|hack|you (?:need|should|must) |make sure|the key|stop doing|start doing"
            r"|number one|ways to|reasons (?:why|to)"
            r"|(?:\d+|two|three|four|five|six|seven|ten) (?:[a-z]+ ){0,2}(?:ways|steps|tips|reasons|rules|things|mistakes|lessons|habits)"
        ),
        "pt": _rx(
            r"como fazer|a melhor (?:forma|maneira)|o (?:melhor )?jeito|dicas?|truques?|passos?|regras?|licao|licoes"
            r"|estrategia|metodo|formula|voce (?:precisa|deve|tem que)|a chave|pare de|comece a"
            r"|numero um|formas de|maneiras de|motivos para|razoes para|aqui (?:esta|estao)"
            r"|(?:\d+|dois|duas|tres|quatro|cinco|seis|sete|dez) (?:[a-z]+ ){0,2}(?:formas|maneiras|jeitos|passos|dicas|motivos|razoes|regras|coisas|erros|licoes|habitos)"
        ),
    },
    "tension": {
        "en": _rx(
            r"but then|until|suddenly|all of a sudden|one day|when i was|i remember|years ago|last (?:year|week|month|night)|the moment"
            r"|out of nowhere|little did|what happened next|ended up|at first|i thought|story"
        ),
        "pt": _rx(
            r"mas ai|so que|ate que|de repente|um dia|quando eu|eu lembro|anos atras|(?:ano|semana|mes) passad[oa]"
            r"|naquele momento|do nada|o que aconteceu|no comeco|eu (?:achava|pensei|pensava)|historia"
        ),
    },
}

# Structured/actionable wording anywhere in a clip (not a hook signal on its own).
STRUCTURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "en": _rx(
        r"first|second|third|step|always|never|instead|for example|remember|important|learn(?:ed)?|you (?:can|have to|need|should|must)"
        r"|the (?:rule|key|lesson|trick)|tip|make sure|stop|start"
    ),
    "pt": _rx(
        r"primeiro|segundo|terceiro|passo|sempre|nunca|em vez de|por exemplo|lembre|importante|aprend(?:i|a|er|eu)"
        r"|voce (?:pode|precisa|deve|tem que)|tem que|a (?:regra|chave|licao)|dica|pare|comece"
    ),
}

PAYOFF_PATTERNS: dict[str, re.Pattern[str]] = {
    "en": _rx(
        r"that'?s why|which is why|the lesson|the point is|the takeaway|in the end|at the end of the day|that'?s (?:how|what|when|the)"
        r"|and that'?s|so remember|the moral|bottom line|ultimately|the result|it worked|this is why|so if you|that changed|turns out"
        r"|the answer|the secret|what matters|the truth is|that is the"
    ),
    "pt": _rx(
        r"por isso|a licao|o ponto e|no final|no fim das contas|e assim|foi assim|foi ai que|entao lembre|moral da historia|resumindo"
        r"|ou seja|o resultado|deu certo|funcionou|e isso que|essa e a|isso mudou|a resposta|o segredo|o que importa|a verdade e"
    ),
}

CONTRAST_PATTERNS: dict[str, re.Pattern[str]] = {
    "en": _rx(r"but|however|instead|actually|until|except|yet|although|turns out|the problem"),
    "pt": _rx(r"mas|porem|so que|em vez|na verdade|ate que|exceto|embora|o problema|entretanto"),
}

BACKREF_PATTERNS: dict[str, re.Pattern[str]] = {
    "en": _rx(
        r"as i said|like i said|as i mentioned|as we (?:saw|discussed|said)|going back to|like you said|you mentioned|as you said"
        r"|the same thing|what (?:he|she|they|you) (?:just )?said|earlier (?:i|we|you)"
    ),
    "pt": _rx(
        r"como eu (?:disse|falei)|como (?:a gente|nos) (?:viu|falou|disse)|como voce (?:disse|falou)|voltando|la atras|o que voce falou"
        r"|mesma coisa|o que (?:ele|ela) (?:acabou de )?(?:falar|dizer|falou|disse)"
    ),
}

SECOND_PERSON: dict[str, re.Pattern[str]] = {
    "en": _rx(r"you|your|you're|you've|you'll"),
    "pt": _rx(r"voce|voces|seu|sua|te|contigo"),
}

QUESTION_STARTERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {
            "why",
            "how",
            "what",
            "who",
            "when",
            "where",
            "which",
            "do",
            "does",
            "did",
            "is",
            "are",
            "can",
            "could",
            "would",
            "should",
            "have",
            "has",
            "will",
        }
    ),
    "pt": frozenset(
        {"por", "como", "qual", "quais", "quem", "quando", "onde", "sera", "voce", "sabe", "sabia"}
    ),
}

_NUMBER_RE = re.compile(r"\d")


def lang_key(language: str) -> str | None:
    """Map a detected language code to a lexicon key (``None`` → language-agnostic mode)."""
    code = (language or "").lower()[:2]
    return code if code in ("en", "pt") else None


@dataclass
class Sentence:
    idx: int
    w0: int  # index of first word (inclusive)
    w1: int  # index after last word (exclusive)
    start: float
    end: float
    text: str
    tokens: list[str] = field(default_factory=list)
    content: list[str] = field(default_factory=list)
    gap_before: float = 0.0
    gap_after: float = 0.0
    is_question: bool = False
    terminal: bool = False

    @property
    def n_words(self) -> int:
        return self.w1 - self.w0

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_sentences(words: list[Word], language: str = "") -> list[Sentence]:
    """Group words into sentence-like units using punctuation and pauses."""
    lk = lang_key(language)
    stop = STOPWORDS.get(lk or "", frozenset())
    out: list[Sentence] = []
    if not words:
        return out
    start_i = 0
    n = len(words)
    for i, w in enumerate(words):
        count = i - start_i + 1
        gap = (words[i + 1].start - w.end) if i + 1 < n else 1e9
        low = w.text.lower()
        terminal = low.endswith(_TERMINAL) and low not in _ABBREV and not re.fullmatch(r"[a-z]\.", low)
        soft = count >= 26 and (low.endswith((",", ";", ":")) or gap > 0.35)
        if terminal or gap > 0.9 or soft or count >= 45 or i == n - 1:
            text = " ".join(x.text for x in words[start_i : i + 1])
            toks = tokenize(text)
            first = toks[0] if toks else ""
            is_q = text.rstrip().endswith(("?", "？")) or (
                lk is not None and first in QUESTION_STARTERS[lk] and text.rstrip().endswith("?")
            )
            out.append(
                Sentence(
                    idx=len(out),
                    w0=start_i,
                    w1=i + 1,
                    start=words[start_i].start,
                    end=w.end,
                    text=text,
                    tokens=toks,
                    content=[t for t in toks if t not in stop and len(t) > 2 and not t.isdigit()],
                    is_question=is_q,
                    terminal=bool(terminal),
                )
            )
            start_i = i + 1
    for k, s in enumerate(out):
        s.gap_before = s.start - out[k - 1].end if k > 0 else s.start
        s.gap_after = out[k + 1].start - s.end if k + 1 < len(out) else 1e9
    return out


class TfIdf:
    """Sentence-level TF-IDF used for similarity, topic boundaries and keyword salience."""

    def __init__(self, docs: list[list[str]], generic: frozenset[str] | None = None) -> None:
        self.n = max(1, len(docs))
        df: Counter[str] = Counter()
        tf: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
            tf.update(d)
        self.idf = {t: math.log((self.n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.tf = tf
        # keywords: recurring, reasonably specific content words
        generic = generic or frozenset()
        scored = [
            (c * min(1.6, self.idf[t] / 2.0), t)
            for t, c in tf.items()
            if c >= 2 and len(t) >= 4 and t not in generic
        ]
        scored.sort(reverse=True)
        self.keywords = [t for _, t in scored[:30]]

    def vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 1.0) for t, c in counts.items()}
        norm_ = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm_ for t, v in vec.items()}

    @staticmethod
    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())


def topic_boundaries(sentences: list[Sentence], tfidf: TfIdf, block: int = 4) -> set[int]:
    """TextTiling-style lexical cohesion: return sentence indices that start a new topic."""
    n = len(sentences)
    if n < max(block * 2 + 1, 30):
        return set()  # too little text for lexical cohesion to be meaningful
    sims: list[float] = []
    for gap in range(1, n):
        left = [t for s in sentences[max(0, gap - block) : gap] for t in s.content]
        right = [t for s in sentences[gap : gap + block] for t in s.content]
        sims.append(TfIdf.cosine(tfidf.vector(left), tfidf.vector(right)) if left and right else 0.0)
    depths: list[float] = []
    for i, s in enumerate(sims):
        lpeak = s
        for j in range(i - 1, -1, -1):
            if sims[j] >= lpeak:
                lpeak = sims[j]
            else:
                break
        rpeak = s
        for j in range(i + 1, len(sims)):
            if sims[j] >= rpeak:
                rpeak = sims[j]
            else:
                break
        depths.append((lpeak - s) + (rpeak - s))
    if not depths:
        return set()
    mean = sum(depths) / len(depths)
    std = math.sqrt(sum((d - mean) ** 2 for d in depths) / len(depths))
    cutoff = mean + 0.5 * std
    return {i + 1 for i, d in enumerate(depths) if d > cutoff and d > 0.1}


def has_number(text: str) -> bool:
    return bool(_NUMBER_RE.search(text))
