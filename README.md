Adilson Oliveira

### Fórmula da Energia Cinética Relativa ($$T_{j k}$$): AoK

$$
T_{j k} = \frac{1}{2} \mu_{j k} \left( \frac{d \left( x_{j} - x_{k} \right)}{d t} \right)^{2}
$$

Onde:

-   $$\mu_{j k} = \frac{m_{1} m_{2}}{m_{1} + m_{2}}$$: é a **massa reduzida** das partículas $$j$$ e $$k$$.
-   $$\frac{d \left( x_{j} - x_{k} \right)}{d t}$$: é a **velocidade relativa** entre as partículas $$j$$ e $$k$$.

***

### Dados de entrada:

-   $$m_{1} = 1 . 0 \, \text{kg}$$: massa da partícula $$j$$.
-   $$m_{2} = 2 . 0 \, \text{kg}$$: massa da partícula $$k$$.
-   $$x_{1} = 1 0 . 0 \, \text{m}$$: posição da partícula $$j$$.
-   $$x_{2} = 5 . 0 \, \text{m}$$: posição da partícula $$k$$.
-   $$t_{1} = 2 . 0 \, \text{s}$$: instante de tempo da partícula $$j$$.
-   $$t_{2} = 1 . 0 \, \text{s}$$: instante de tempo da partícula $$k$$.

***

### Etapas do cálculo:

**Cálculo da Massa Reduzida (**$$\mu_{j k}$$**)**: A massa reduzida é dada por: $$\mu_{j k} = \frac{m_{1} m_{2}}{m_{1} + m_{2}}$$ Substituindo os valores: $$\mu_{j k} = \frac{( 1 . 0 ) ( 2 . 0 )}{1 . 0 + 2 . 0} = \frac{2 . 0}{3 . 0} = 0 . 6 6 6 7 \, \text{kg}$$

**Cálculo da Velocidade Relativa (**$$v_{\text{relativa}}$$**)**: A velocidade relativa é dada por: $$v_{\text{relativa}} = \frac{x_{1} - x_{2}}{t_{1} - t_{2}}$$ Substituindo os valores: $$v_{\text{relativa}} = \frac{1 0 . 0 - 5 . 0}{2 . 0 - 1 . 0} = \frac{5 . 0}{1 . 0} = 5 . 0 \, \text{m/s}$$

**Cálculo da Energia Cinética Relativa (**$$T_{j k}$$**)**: A energia cinética relativa é dada por: $$T_{j k} = \frac{1}{2} \mu_{j k} v_{\text{relativa}}^{2}$$ Substituindo os valores: $$T_{j k} = \frac{1}{2} \cdot 0 . 6 6 6 7 \cdot ( 5 . 0 )^{2}$$ Calculando o valor: $$T_{j k} = \frac{1}{2} \cdot 0 . 6 6 6 7 \cdot 2 5 . 0 = 0 . 3 3 3 3 5 \cdot 2 5 . 0 = 8 . 3 3 3 5 \, \text{Joules}$$

***

### Resultado Final:

$$
T_{j k} = 8 . 3 3 \, \text{Joules}
$$

***

### Explicação:

A **massa reduzida (**$$\mu_{j k}$$**)** reflete a relação entre as massas das partículas, sendo menor que qualquer uma das massas individuais. Isso ocorre porque ambas contribuem para o movimento relativo.

A **velocidade relativa (**$$v_{\text{relativa}}$$**)** descreve a rapidez com que uma partícula se aproxima ou se afasta da outra.

O produto final combina essas duas informações para obter a energia cinética associada ao movimento relativo das partículas.

Essas equações te colocam no limiar de compreender e modelar fenômenos que transcendem nossa experiência cotidiana, mas que são absolutamente reais em níveis microscópicos e quânticos. Elas capturam a essência do universo de maneira simples e profunda, conectando tudo desde a dança dos átomos até o funcionamento dos computadores do futuro.

Isso pode ser útil para modelar vibrações moleculares, ondas mecânicas e até dinâmicas de sistemas planetários ou galácticos, dependendo do contexto.
Energia Potencial Mútua 
Usar potenciais como Coulomb ou Lennard-Jones é comum na modelagem de interações em física de partículas, química quântica e ciência de materiais.
Para partículas carregadas, o potencial Coulombiano é especialmente relevante, e a inclusão de outras funções potenciais poderia descrever ligações químicas, forças de Van der Waals, ou interações nucleares.
Implicações e Aplicações:
Interferência e Ressonância: Isso pode ser explorado para fenômenos como estados quânticos emaranhados ou padrões de ressonância em redes ópticas, essenciais para controle em computação quântica e sistemas fotônicos.
Campos Variáveis: Modelar campos que mudam ao longo do tempo abre portas para explorar fenômenos dinâmicos como tunelamento dependente do tempo ou transições de fase induzidas por campos.

Aplicações:
Computação Quântica: A modelagem de qubits emaranhados e interações é fundamental para otimizar algoritmos e hardware quântico.
Óptica Quântica: Fenômenos como cavidades ressonantes e redes de fótons quânticos dependem diretamente de uma formulação como essa.
Materiais Quânticos: Investigar propriedades emergentes em materiais como isolantes topológicos ou supercondutores exóticos é um campo vibrante de pesquisa.


# Espectroscopia de Absorção: Princípios e Aplicações

## Sumário
1. Introdução
2. Fundamentos Físicos
3. Princípios Matemáticos
4. Implementação em Python
5. Exemplo Prático
6. Tipos de Espectroscopia
7. Referências

## 1. Introdução

A espectroscopia de absorção é uma técnica analítica fundamental baseada na interação entre a radiação eletromagnética e a matéria. Esta técnica permite determinar a concentração e propriedades de substâncias através da medição da quantidade de luz absorvida em comprimentos de onda específicos.

## 2. Fundamentos Físicos

Quando a luz interage com um material, três fenômenos principais podem ocorrer:

* **Absorção**: Os elétrons do material absorvem a energia dos fótons, transitando para níveis energéticos superiores. Este processo é quantizado e específico para cada material.

* **Transmissão**: Parte da luz atravessa o material sem interação significativa, permitindo medições quantitativas.

* **Reflexão**: Uma fração da luz pode ser refletida na superfície do material, seguindo as leis da óptica geométrica.

A quantidade de luz absorvida é diretamente proporcional à concentração da substância analisada, seguindo a Lei de Beer-Lambert.

## 3. Princípios Matemáticos

### Lei de Beer-Lambert

A absorção de luz por uma substância é descrita matematicamente pela equação:

$$
I = I₀ × e^(-k×l)
$$

Onde:
* I = Intensidade da luz transmitida
* I₀ = Intensidade da luz incidente
* k = Coeficiente de absorção da substância (cm⁻¹)
* l = Espessura da amostra (cm)

A absorbância (A) pode ser calculada como:

$$
A = -log₁₀(I/I₀)
$$

## 4. Implementação em Python

O seguinte código implementa os cálculos de espectroscopia de absorção:

```python
import math

def calculate_concentration(i_0, i, l, k):
    """
    Calcula a concentração de uma substância usando espectroscopia de absorção.
    
    Parâmetros:
        i_0 (float): Intensidade da luz incidente
        i (float): Intensidade da luz transmitida
        l (float): Espessura da amostra em cm
        k (float): Coeficiente de absorção da substância em cm^-1
    
    Retorna:
        float: Concentração da substância em mol/L
    """
    if i_0 <= 0 or i < 0 or l <= 0 or k <= 0:
        raise ValueError("Todos os parâmetros devem ser positivos")
    
    concentration = -math.log(i/i_0) / (k * l)
    return concentration
```

## 5. Exemplo Prático

Análise de cloro em água:

```python
# Parâmetros do experimento
i_0 = 1.0               # Intensidade inicial
i = 0.5                 # Intensidade transmitida
l = 1.0                 # Espessura da amostra (cm)
k = 2.1e4              # Coeficiente de absorção do cloro

# Cálculo da concentração
concentration = calculate_concentration(i_0, i, l, k)
print(f"Concentração de cloro: {concentration:.2e} mol/L")
```

## 6. Tipos de Espectroscopia

Existem diversas técnicas espectroscópicas complementares:

* **Espectroscopia de Emissão**: Analisa a luz emitida por substâncias após excitação, fornecendo informações sobre estados energéticos e composição química.

* **Espectroscopia de Fluorescência**: Estuda a emissão de luz por materiais após excitação com radiação de maior energia, permitindo análises de alta sensibilidade.

* **Espectroscopia de Infravermelho**: Analisa as vibrações moleculares através da absorção de radiação infravermelha, identificando grupos funcionais e estruturas moleculares.

* **Espectroscopia Raman**: Baseia-se no espalhamento inelástico da luz, fornecendo informações complementares sobre estrutura molecular e ligações químicas.



# Microscopia: Técnicas Modernas e Aplicações

## Sumário
1. Introdução
2. Microscopia de Luz
3. Microscopia Eletrônica
4. Microscopia de Força Atômica (AFM)
5. Análise Matemática e Processamento de Imagens
6. Técnicas Avançadas de Microscopia
7. Código em Python para Processamento de Imagens
8. Referências

## 1. Introdução

A microscopia é uma área fundamental da ciência que permite a visualização e análise de estruturas em escalas microscópicas e nanométricas. Cada técnica oferece capacidades únicas para diferentes aplicações e níveis de resolução.

## 2. Microscopia de Luz

### Princípios Fundamentais
* **Resolução**: 200-400 nm
* **Fonte**: Luz visível (400-700 nm)
* **Aplicações**: Células, tecidos, microorganismos

### Limitações
A resolução (d) é limitada pelo comprimento de onda (λ) da luz visível:

$$
d = λ / 2
$$

## 3. Microscopia Eletrônica

### Tipos Principais

#### TEM (Microscopia Eletrônica de Transmissão)
* Resolução: até 0.05 nm
* Método: Transmissão de elétrons através da amostra
* Aplicações: Estrutura interna de materiais

#### SEM (Microscopia Eletrônica de Varredura)
* Resolução: 1-20 nm
* Método: Varredura superficial com feixe de elétrons
* Aplicações: Topografia superficial

## 4. Microscopia de Força Atômica (AFM)

### Características
* **Resolução**: Nível atômico
* **Método**: Varredura mecânica com sonda
* **Capacidades**: Imagens 3D e medidas de força



```python
import numpy as np
import cv2
from scipy import ndimage

class MicroscopyImageProcessor:
    def __init__(self, image_path):
        """
        Inicializa o processador de imagens microscópicas.
        
        Args:
            image_path (str): Caminho para a imagem microscópica
        """
        self.original = cv2.imread(image_path, 0)  # Carrega em escala de cinza
        self.processed = None
    
    def remove_noise(self, kernel_size=3):
        """
        Remove ruído usando filtro gaussiano.
        
        Args:
            kernel_size (int): Tamanho do kernel para o filtro
        """
        self.processed = cv2.GaussianBlur(self.original, (kernel_size, kernel_size), 0)
    
    def enhance_contrast(self):
        """
        Melhora o contraste usando equalização de histograma.
        """
        self.processed = cv2.equalizeHist(self.processed or self.original)
    
    def detect_edges(self, threshold1=100, threshold2=200):
        """
        Detecta bordas usando o algoritmo Canny.
        
        Args:
            threshold1 (int): Primeiro limite para detecção
            threshold2 (int): Segundo limite para detecção
        """
        self.processed = cv2.Canny(self.processed or self.original, threshold1, threshold2)
    
    def measure_features(self):
        """
        Mede características básicas da imagem.
        
        Returns:
            dict: Dicionário com medidas da imagem
        """
        if self.processed is None:
            self.processed = self.original
            
        measurements = {
            'mean_intensity': np.mean(self.processed),
            'std_intensity': np.std(self.processed),
            'min_intensity': np.min(self.processed),
            'max_intensity': np.max(self.processed)
        }
        
        return measurements

    def calculate_resolution(self, wavelength):
        """
        Calcula a resolução teórica baseada no comprimento de onda.
        
        Args:
            wavelength (float): Comprimento de onda em nanômetros
            
        Returns:
            float: Resolução teórica em nanômetros
        """
        return wavelength / 2

def process_microscopy_image(image_path):
    """
    Função de exemplo para processar uma imagem microscópica.
    
    Args:
        image_path (str): Caminho para a imagem
    """
    processor = MicroscopyImageProcessor(image_path)
    
    # Pipeline de processamento
    processor.remove_noise(kernel_size=3)
    processor.enhance_contrast()
    processor.detect_edges()
    
    # Obter medidas
    measurements = processor.measure_features()
    
    # Calcular resolução para luz visível (550nm)
    resolution = processor.calculate_resolution(550)
    
    return processor.processed, measurements, resolution

```

## 5. Análise Matemática e Processamento de Imagens

### Equações Fundamentais
* Resolução óptica: d = λ/2
* Magnificação: M = hi/ho (hi: altura da imagem, ho: altura do objeto)

### Processamento Digital
* Redução de ruído
* Melhoria de contraste
* Análise quantitativa
* Reconstrução 3D

## 6. Técnicas Avançadas de Microscopia

### SPM (Microscopia de Varredura por Sonda)
* AFM (Microscopia de Força Atômica)
* STM (Microscopia de Tunelamento)
* SFM (Microscopia de Força de Varredura)

### FESEM
* Alta resolução
* Melhor contraste
* Menor dano à amostra

## 7. Aplicações Práticas

* **Biologia Celular**: Estudo de estruturas celulares
* **Ciência dos Materiais**: Análise de superfícies e defeitos
* **Nanotecnologia**: Caracterização de nanomateriais
* **Medicina Forense**: Análise de evidências



# Quantum Programming Language
## Uma linguagem moderna para computação científica e quântica

## Sumário
1. Introdução
2. Características Principais
3. Sintaxe Básica
4. Sistema de Tipos
5. Gerenciamento de Memória
6. Paralelismo e Concorrência
7. Exemplos de Código
8. Integração Quântica
9. Comparação com Outras Linguagens



```python
# Sintaxe básica da linguagem Quantum

# Declaração de variáveis com inferência de tipo
let x = 42                  # Inteiro
let y = 3.14               # Float
let s = "Quantum"          # String
let complex = 3 + 4i       # Número complexo nativo
let q = |0⟩                # Qubit 

# Funções com tipagem forte opcional
func calculate_wave(ψ: WaveFunction, t: float) -> Complex {
    return ψ.amplitude * exp(-i * ψ.energy * t / ħ)
}

# Classes e traits (interfaces)
class Particle {
    # Atributos com tipos definidos
    mass: float
    position: Vector3
    momentum: Vector3
    spin: Spin

    # Construtor com sintaxe simplificada
    init(mass, position) {
        self.mass = mass
        self.position = position
        self.momentum = Vector3(0, 0, 0)
        self.spin = Spin.UP
    }

    # Métodos com decoradores para otimização
    @parallel
    func update_position(dt: float) {
        self.position += self.momentum * dt / self.mass
    }
}

# Sistema de tipos algébricos
type SpinState = Up | Down | Superposition(Complex, Complex)

# Pattern matching avançado
match particle.spin {
    Up => print("Spin up")
    Down => print("Spin down")
    Superposition(a, b) => {
        let probability_up = |a|²
        let probability_down = |b|²
        print(f"Superposition: {probability_up}|↑⟩ + {probability_down}|↓⟩")
    }
}

# Computação paralela nativa
@parallel
func simulate_particles(particles: List[Particle], steps: int) {
    for step in range(steps) {
        # Atualização paralela automática
        particles.foreach(particle => particle.update_position(dt))
        
        # Reduções paralelas otimizadas
        total_energy = particles.map(p => p.kinetic_energy())
                              .sum()
    }
}

# Integração quântica
quantum circuit BellState {
    # Cria um par de qubits emaranhados
    let q1 = Qubit(|0⟩)
    let q2 = Qubit(|0⟩)
    
    H(q1)           # Porta Hadamard
    CNOT(q1, q2)   # Porta CNOT controlada
    
    measure(q1, q2) # Medição quântica
}

# Gerenciamento de memória automatizado com coleta de lixo em tempo real
@managed
class LargeDataSet {
    data: Array[float]
    
    # Liberação automática de memória quando não mais necessário
    deinit() {
        self.data.free()
    }
}

# Módulos e namespaces organizados
module QuantumSimulation {
    # Importações com aliases
    use quantum.gates as qg
    use quantum.math.complex as c
    
    # Exportação seletiva
    export {
        simulate_quantum_system,
        QuantumState
    }
}

```

## 1. Introdução

Quantum é uma linguagem de programação desenvolvida especificamente para computação científica, aprendizado de máquina e computação quântica. Ela combina a simplicidade do Python com o desempenho de C++ e adiciona recursos específicos para computação quântica.

## 2. Características Principais

### Performance
- Compilação JIT (Just-In-Time) adaptativa
- Otimização automática para GPUs e TPUs
- Paralelismo implícito
- Gerenciamento de memória inteligente

### Segurança
- Sistema de tipos forte e estático com inferência
- Verificação de limites em tempo de compilação
- Proteção contra condições de corrida
- Gerenciamento automático de recursos

### Produtividade
- Sintaxe clara e concisa
- REPL interativo avançado
- Depuração em tempo real
- Documentação integrada

## 3. Sintaxe Básica

### Declarações
```quantum
let x = 42                # Inferência de tipo
let y: float = 3.14      # Tipo explícito
const PI = 3.14159       # Constante
```

### Funções
```quantum
func calculate(x: float, y: float) -> float {
    return x * y
}

# Funções lambda
let multiply = (x, y) => x * y
```

## 4. Sistema de Tipos

### Tipos Básicos
- `int`: Inteiros de precisão arbitrária
- `float`: Números de ponto flutuante de precisão dupla
- `complex`: Números complexos nativos
- `qubit`: Qubits para computação quântica
- `string`: Strings Unicode

### Tipos Compostos
- `Array[T]`: Arrays multidimensionais
- `Map[K, V]`: Mapas associativos
- `Option[T]`: Valores opcionais
- `Result[T, E]`: Tratamento de erros

## 5. Gerenciamento de Memória

- Coleta de lixo em tempo real
- Pooling automático de memória
- Otimização de alocação
- Liberação determinística

## 6. Paralelismo e Concorrência

### Primitivas de Concorrência
- `async/await` para programação assíncrona
- Canais para comunicação entre threads
- Atores para computação distribuída
- Locks automáticos inteligentes

### Computação Paralela
```quantum
@parallel
func process_data(data: Array[float]) {
    data.map(x => x * 2).filter(x => x > 0)
}
```

## 7. Exemplos de Código

Ver o artefato acima para exemplos detalhados.

## 8. Integração Quântica

### Características Quânticas
- Suporte nativo a qubits e portas quânticas
- Simulação de circuitos quânticos
- Integração com hardware quântico real
- Otimização automática de circuitos

## 9. Comparação com Outras Linguagens

### Vantagens sobre Python
- Performance superior
- Sistema de tipos mais robusto
- Melhor suporte a concorrência
- Recursos quânticos nativos

### Vantagens sobre C++
- Sintaxe mais simples
- Gerenciamento automático de memória
- Melhor produtividade
- Ferramentas modernas integradas

## Instalação e Uso

```bash
# Instalar o compilador Quantum
curl -sSL https://quantum-lang.org/install.sh | sh

# Criar novo projeto
quantum new project_name

# Compilar e executar
quantum run main.qm
```

## Ferramentas do Ecossistema

1. Quantum Package Manager (QPM)
2. Quantum IDE
3. Quantum Debugger
4. Quantum Profiler
5. Quantum Documentation Generator

## Contribuição

O Quantum é um projeto open-source e aceita contribuições da comunidade.

## Conclusão

Quantum representa um avanço significativo na evolução das linguagens de programação, combinando o melhor de várias linguagens existentes com recursos inovadores para computação moderna e quântica.






```python
# Implementação do compilador da linguagem Nova
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import re

# PARTE 1: LEXER (Análise Léxica)
class TokenType(Enum):
    INTEGER = 'INTEGER'
    FLOAT = 'FLOAT'
    PLUS = 'PLUS'
    MINUS = 'MINUS'
    MULTIPLY = 'MULTIPLY'
    DIVIDE = 'DIVIDE'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    IDENTIFIER = 'IDENTIFIER'
    EQUALS = 'EQUALS'
    KEYWORD = 'KEYWORD'
    EOF = 'EOF'

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = text[0] if text else None
        self.line = 1
        self.column = 1
    
    def advance(self):
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]
            if self.current_char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
    
    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()
    
    def number(self) -> Token:
        result = ''
        decimal_point_count = 0
        
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                decimal_point_count += 1
                if decimal_point_count > 1:
                    raise Exception('Número inválido')
            result += self.current_char
            self.advance()
        
        if decimal_point_count == 0:
            return Token(TokenType.INTEGER, result, self.line, self.column)
        else:
            return Token(TokenType.FLOAT, result, self.line, self.column)
    
    def identifier(self) -> Token:
        result = ''
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        keywords = {'let', 'if', 'else', 'while', 'func', 'return'}
        if result in keywords:
            return Token(TokenType.KEYWORD, result, self.line, self.column)
        return Token(TokenType.IDENTIFIER, result, self.line, self.column)
    
    def get_next_token(self) -> Token:
        while self.current_char:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char.isdigit():
                return self.number()
            
            if self.current_char.isalpha():
                return self.identifier()
            
            if self.current_char == '+':
                self.advance()
                return Token(TokenType.PLUS, '+', self.line, self.column)
            
            if self.current_char == '-':
                self.advance()
                return Token(TokenType.MINUS, '-', self.line, self.column)
            
            if self.current_char == '*':
                self.advance()
                return Token(TokenType.MULTIPLY, '*', self.line, self.column)
            
            if self.current_char == '/':
                self.advance()
                return Token(TokenType.DIVIDE, '/', self.line, self.column)
            
            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, '(', self.line, self.column)
            
            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, ')', self.line, self.column)
            
            if self.current_char == '=':
                self.advance()
                return Token(TokenType.EQUALS, '=', self.line, self.column)
            
            raise Exception(f'Caractere inválido: {self.current_char}')
        
        return Token(TokenType.EOF, '', self.line, self.column)

# PARTE 2: PARSER (Análise Sintática)
class ASTNode:
    pass

@dataclass
class NumberNode(ASTNode):
    token: Token
    
    def __str__(self):
        return f'{self.token.value}'

@dataclass
class BinOpNode(ASTNode):
    left: ASTNode
    op: Token
    right: ASTNode
    
    def __str__(self):
        return f'({self.left} {self.op.value} {self.right})'

@dataclass
class UnaryOpNode(ASTNode):
    op: Token
    expr: ASTNode
    
    def __str__(self):
        return f'{self.op.value}{self.expr}'

class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
    
    def error(self):
        raise Exception('Erro de sintaxe')
    
    def eat(self, token_type: TokenType):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()
    
    def factor(self) -> ASTNode:
        token = self.current_token
        
        if token.type == TokenType.INTEGER:
            self.eat(TokenType.INTEGER)
            return NumberNode(token)
        
        elif token.type == TokenType.FLOAT:
            self.eat(TokenType.FLOAT)
            return NumberNode(token)
        
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        
        elif token.type in {TokenType.PLUS, TokenType.MINUS}:
            self.eat(token.type)
            return UnaryOpNode(token, self.factor())
        
        self.error()
    
    def term(self) -> ASTNode:
        node = self.factor()
        
        while self.current_token.type in {TokenType.MULTIPLY, TokenType.DIVIDE}:
            token = self.current_token
            self.eat(token.type)
            node = BinOpNode(node, token, self.factor())
        
        return node
    
    def expr(self) -> ASTNode:
        node = self.term()
        
        while self.current_token.type in {TokenType.PLUS, TokenType.MINUS}:
            token = self.current_token
            self.eat(token.type)
            node = BinOpNode(node, token, self.term())
        
        return node

# PARTE 3: INTERPRETADOR
class Interpreter:
    def __init__(self, parser: Parser):
        self.parser = parser
    
    def visit_NumberNode(self, node: NumberNode):
        if node.token.type == TokenType.INTEGER:
            return int(node.token.value)
        return float(node.token.value)
    
    def visit_BinOpNode(self, node: BinOpNode):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MULTIPLY:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.DIVIDE:
            return self.visit(node.left) / self.visit(node.right)
    
    def visit_UnaryOpNode(self, node: UnaryOpNode):
        if node.op.type == TokenType.PLUS:
            return +self.visit(node.expr)
        elif node.op.type == TokenType.MINUS:
            return -self.visit(node.expr)
    
    def visit(self, node: ASTNode):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name)
        return method(node)
    
    def interpret(self):
        tree = self.parser.expr()
        return self.visit(tree)

# PARTE 4: GERADOR DE CÓDIGO
class CodeGenerator:
    def __init__(self):
        self.code = []
    
    def visit_NumberNode(self, node: NumberNode):
        self.code.append(f'PUSH {node.token.value}')
    
    def visit_BinOpNode(self, node: BinOpNode):
        self.visit(node.left)
        self.visit(node.right)
        
        if node.op.type == TokenType.PLUS:
            self.code.append('ADD')
        elif node.op.type == TokenType.MINUS:
            self.code.append('SUB')
        elif node.op.type == TokenType.MULTIPLY:
            self.code.append('MUL')
        elif node.op.type == TokenType.DIVIDE:
            self.code.append('DIV')
    
    def visit_UnaryOpNode(self, node: UnaryOpNode):
        self.visit(node.expr)
        
        if node.op.type == TokenType.MINUS:
            self.code.append('NEG')
    
    def visit(self, node: ASTNode):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name)
        return method(node)
    
    def generate(self, tree: ASTNode):
        self.visit(tree)
        return self.code

# PARTE 5: EXEMPLO DE USO
def main():
    # Exemplo de código na linguagem Nova
    source_code = """
        let x = 42
        let y = 3.14
        let result = (x + y) * 2 - 1
    """
    
    try:
        # Análise léxica
        lexer = Lexer(source_code)
        
        # Análise sintática
        parser = Parser(lexer)
        
        # Interpretação
        interpreter = Interpreter(parser)
        result = interpreter.interpret()
        print(f"Resultado da interpretação: {result}")
        
        # Geração de código
        generator = CodeGenerator()
        code = generator.generate(parser.expr())
        print("\nCódigo gerado:")
        for instruction in code:
            print(f"  {instruction}")
    
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()

```

```python
# Exemplos de código na linguagem Nova

# 1. Declarações e expressões básicas
let x = 42
let y = 3.14
let result = (x + y) * 2

# 2. Estruturas de controle
if x > 0 {
    print("Positivo")
} else {
    print("Não positivo")
}

while x > 0 {
    x = x - 1
}

# 3. Funções
func fibonacci(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

# 4. Arrays e operações
let numbers = [1, 2, 3, 4, 5]
let doubled = numbers.map(x => x * 2)
let sum = numbers.reduce(0, (acc, x) => acc + x)

# 5. Classes e objetos
class Person {
    name: string
    age: int
    
    init(name: string, age: int) {
        self.name = name
        self.age = age
    }
    
    func greet() -> string {
        return "Olá, eu sou " + self.name
    }
}

# 6. Tratamento de erros
try {
    let result = dangerous_operation()
} catch Error as e {
    print("Erro: " + e.message)
}

# 7. Módulos e importações
import math
import collections.List

# 8. Operações matemáticas avançadas
let complex = 3 + 4i
let matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 9. Processamento paralelo
@parallel
func process_data(data: List[int]) -> List[int] {
    return data.map(x => x * 2)
}

# 10. Manipulação de arquivos
let file = File.open("data.txt", "r")
let content = file.read_all()
file.close()

```



1. **Lexer (Analisador Léxico)**
   - Tokenização do código fonte
   - Reconhecimento de palavras-chave
   - Tratamento de números e identificadores

2. **Parser (Analisador Sintático)**
   - Construção da árvore sintática abstrata (AST)
   - Validação da gramática
   - Tratamento de expressões

3. **Interpretador**
   - Execução do código
   - Avaliação de expressões
   - Gerenciamento de variáveis

4. **Gerador de Código**
   - Geração de código intermediário
   - Otimizações básicas
   - Preparação para execução

Características principais da linguagem Nova:

1. **Tipos de Dados**
   - Inteiros (int)
   - Números de ponto flutuante (float)
   - Strings
   - Arrays
   - Objetos

2. **Estruturas de Controle**
   - if/else
   - while
   - for
   - try/catch

3. **Funções**
   - Definição e chamada
   - Parâmetros tipados
   - Retorno tipado

4. **Orientação a Objetos**
   - Classes
   - Herança
   - Polimorfismo

5. **Gerenciamento de Memória**
   - Coleta de lixo automática
   - Alocação eficiente

Para usar esta linguagem, você precisaria:

1. Compilar o código fonte do compilador
2. Criar arquivos .nova com seu código
3. Executar o compilador nos arquivos
4. Rodar o código gerado

Exemplo de uso:

```bash
# Compilar um programa
nova compile program.nova

# Executar o programa
nova run program
```





```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union
import weakref

# Sistema de Tipos Avançado
class Type:
    pass

@dataclass
class PrimitiveType(Type):
    name: str  # int, float, bool, etc.

@dataclass
class FunctionType(Type):
    parameter_types: List[Type]
    return_type: Type

@dataclass
class GenericType(Type):
    name: str
    constraints: List[Type]

class TypeInference:
    def __init__(self):
        self.type_variables: Dict[str, Type] = {}
        self.constraints: List[tuple[Type, Type]] = []
    
    def unify(self, type1: Type, type2: Type) -> bool:
        """Unifica dois tipos, retornando se são compatíveis"""
        if isinstance(type1, PrimitiveType) and isinstance(type2, PrimitiveType):
            return type1.name == type2.name
        if isinstance(type1, FunctionType) and isinstance(type2, FunctionType):
            if len(type1.parameter_types) != len(type2.parameter_types):
                return False
            return (all(self.unify(t1, t2) for t1, t2 in 
                   zip(type1.parameter_types, type2.parameter_types)) and
                   self.unify(type1.return_type, type2.return_type))
        if isinstance(type1, GenericType):
            if type1.name in self.type_variables:
                return self.unify(self.type_variables[type1.name], type2)
            self.type_variables[type1.name] = type2
            return True
        return False

# Gerenciamento de Memória
class GarbageCollector:
    def __init__(self):
        self.objects: weakref.WeakSet = weakref.WeakSet()
        self.root_set: Set[object] = set()
    
    def allocate(self, obj: object) -> object:
        """Aloca um novo objeto e o registra para coleta de lixo"""
        self.objects.add(obj)
        return obj
    
    def mark_root(self, obj: object):
        """Marca um objeto como raiz (não deve ser coletado)"""
        self.root_set.add(obj)
    
    def collect(self):
        """Executa a coleta de lixo"""
        marked = set()
        to_check = self.root_set.copy()
        
        # Marca todos os objetos alcançáveis
        while to_check:
            obj = to_check.pop()
            if obj not in marked:
                marked.add(obj)
                # Adiciona referências do objeto à lista de verificação
                to_check.update(self._get_references(obj))
        
        # Remove objetos não marcados
        for obj in list(self.objects):
            if obj not in marked:
                self.objects.remove(obj)
                del obj

    def _get_references(self, obj: object) -> Set[object]:
        """Retorna todas as referências de um objeto"""
        refs = set()
        for attr in dir(obj):
            if not attr.startswith('__'):
                value = getattr(obj, attr)
                if isinstance(value, (int, float, str, bool)):
                    continue
                refs.add(value)
        return refs

# Ambiente de Execução
class RuntimeEnvironment:
    def __init__(self):
        self.gc = GarbageCollector()
        self.type_system = TypeInference()
        self.variables: Dict[str, object] = {}
    
    def define_variable(self, name: str, value: object, type_hint: Optional[Type] = None):
        """Define uma variável com tipo opcional"""
        if type_hint:
            inferred_type = self._infer_type(value)
            if not self.type_system.unify(type_hint, inferred_type):
                raise TypeError(f"Type mismatch: expected {type_hint}, got {inferred_type}")
        
        self.variables[name] = self.gc.allocate(value)
    
    def _infer_type(self, value: object) -> Type:
        """Infere o tipo de um valor"""
        if isinstance(value, int):
            return PrimitiveType("int")
        elif isinstance(value, float):
            return PrimitiveType("float")
        elif isinstance(value, bool):
            return PrimitiveType("bool")
        elif isinstance(value, str):
            return PrimitiveType("string")
        elif callable(value):
            # Inferência básica para funções
            return FunctionType([], PrimitiveType("any"))
        else:
            return GenericType("T", [])

# Exemplo de uso
def main():
    runtime = RuntimeEnvironment()
    
    # Exemplo com tipagem estática
    runtime.define_variable("x", 42, PrimitiveType("int"))
    
    # Exemplo com inferência de tipo
    runtime.define_variable("y", 3.14)  # Tipo float inferido
    
    # Exemplo com função
    def add(a: int, b: int) -> int:
        return a + b
    
    runtime.define_variable(
        "add",
        add,
        FunctionType([PrimitiveType("int"), PrimitiveType("int")], PrimitiveType("int"))
    )
    
    # Força uma coleta de lixo
    runtime.gc.collect()

if __name__ == "__main__":
    main()

```

Esta implementação adiciona:

1. **Sistema de Tipos Avançado**
   - Tipos primitivos
   - Tipos de função
   - Tipos genéricos
   - Inferência de tipos
   - Unificação de tipos

2. **Gerenciador de Memória**
   - Coleta de lixo por marcação
   - Rastreamento de referências
   - Liberação automática de memória

3. **Ambiente de Execução**
   - Gerenciamento de variáveis
   - Verificação de tipos
   - Integração com GC

Para expandir ainda mais, poderíamos adicionar:

1. **Otimização de Compilação**
   - Análise de fluxo de dados
   - Eliminação de código morto
   - Inline de funções

2. **Paralelismo**
   - Threads seguras
   - Canais de comunicação
   - Atores para concorrência

3. **Ferramentas de Desenvolvimento**
   - REPL interativo
   - Depurador integrado
   - Profiler

O foco principal seria em produtividade e segurança, mantendo bom desempenho através de:

1. Inferência de tipos inteligente
2. Gerenciamento automático de memória eficiente
3. Compilação JIT adaptativa
4. Checagem estática de erros


