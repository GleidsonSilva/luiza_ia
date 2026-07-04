JOGOS_CONHECIDOS = {
    "forca": {
        "nome": "Forca",
        "descricao": "Eu penso em uma palavra e voce tenta adivinhar as letras. Cada erro ganha uma parte do bonequinho na forca!",
        "instrucoes": "Escolha uma letra por vez. Tente descobrir a palavra antes do boneco ser completo.",
    },
    "adivinhacao": {
        "nome": "Adivinhacao",
        "descricao": "Eu penso em algo (animal, comida, objeto) e voce faz perguntas que eu respondo so com sim ou nao ate descobrir!",
        "instrucoes": "Faca perguntas que eu possa responder com sim ou nao. Tenta descobrir o que eu pensei.",
    },
    "20 perguntas": {
        "nome": "20 Perguntas",
        "descricao": "Eu penso em um personagem ou objeto e voce tem 20 perguntas para descobrir o que eh!",
        "instrucoes": "Faca perguntas de sim ou nao. Voce tem 20 chances para acertar!",
    },
    "simao diz": {
        "nome": "Simao Diz",
        "descricao": "Eu sou o Simao! Faco acoes e voce so deve fazer se eu falar 'Simao diz' primeiro.",
        "instrucoes": "Preste atencao! So obedeeca se eu disser 'Simao diz' antes da acao.",
    },
    "simão diz": {
        "nome": "Simão Diz",
        "descricao": "Eu sou o Simão! Faço ações e você só deve fazer se eu falar 'Simão diz' primeiro.",
        "instrucoes": "Preste atenção! Só obedeça se eu disser 'Simão diz' antes da ação.",
    },
    "historia coletiva": {
        "nome": "Historia Coletiva",
        "descricao": "A gente cria uma historia juntos! Eu falo uma frase, voce continua, e assim vamos montando uma historia bem legal.",
        "instrucoes": "Continue a historia de onde eu parei. Use sua imaginacao!",
    },
    "deteive": {
        "nome": "Detetive ou Mistério",
        "descricao": "Eu invento um misterio bem simples e voce faz perguntas para descobrir o que aconteceu!",
        "instrucoes": "Investigue fazendo perguntas. Descubra o culpado, o lugar e o motivo!",
    },
    "palavra secreta": {
        "nome": "Palavra Secreta",
        "descricao": "Eu dou dicas sobre uma palavra secreta e voce tenta adivinhar qual eh!",
        "instrucoes": "Use as dicas para descobrir a palavra secreta. Quanto menos dicas voce precisar, melhor!",
    },
    "batalha naval": {
        "nome": "Batalha Naval (simplificada)",
        "descricao": "A gente brinca de batalha naval simplificada! Cada um tem navios escondidos em um tabuleiro e tenta acertar os do outro.",
        "instrucoes": "Escolha uma posicao para atirar. Tente acertar todos os meus navios!",
    },
    "caça ao tesouro": {
        "nome": "Caça ao Tesouro",
        "descricao": "Eu dou pistas e voce segue ate encontrar o tesouro escondido!",
        "instrucoes": "Siga as pistas que eu vou dar. Cada pista leva a proxima ate o tesouro!",
    },
}

def listar_jogos() -> str:
    nomes = [v["nome"] for v in JOGOS_CONHECIDOS.values()]
    unicos = list(dict.fromkeys(nomes))
    return ", ".join(unicos)

def descricao_jogo(nome: str) -> str | None:
    for chave, jogo in JOGOS_CONHECIDOS.items():
        if nome.lower() in chave or nome.lower() in jogo["nome"].lower():
            return f"{jogo['nome']}: {jogo['descricao']}"
    return None
