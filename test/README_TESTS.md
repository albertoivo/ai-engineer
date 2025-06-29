# Testes Unitários - OpenAI Service

Este arquivo contém testes unitários completos para o módulo `openai_service.py`, seguindo as melhores práticas de teste para APIs externas.

## Estrutura dos Testes

### 1. **Mocking da API OpenAI**
- Testa formatos de entrada/saída
- Verifica parâmetros enviados à API
- Simula respostas sem fazer chamadas reais

### 2. **Validação de Parâmetros**
- Testa entradas válidas e inválidas
- Verifica tratamento de valores None/vazios
- Valida limites de tokens e modelos

### 3. **Tratamento de Erros**
- Simula erros da API (RateLimitError, APIError)
- Testa erros de conexão
- Verifica respostas mal formadas

### 4. **Integridade Funcional**
- Verifica consistência entre funções similares
- Testa configurações de temperatura apropriadas
- Valida processamento de dados

## Como Executar

### Método 1: Script Automatizado
```bash
chmod +x run_tests.sh
./run_tests.sh
```

### Método 2: Unittest Direto
```bash
python -m unittest test_openai_service.py -v
```

### Método 3: Com Coverage (após instalar)
```bash
pip install coverage
coverage run -m unittest test_openai_service.py
coverage report
coverage html  # Gera relatório HTML
```

### Método 4: Com Pytest (mais recursos)
```bash
pip install -r requirements-test.txt
pytest test_openai_service.py -v
pytest test_openai_service.py --cov=openai_service --cov-report=html
```

## Principais Testes Implementados

### Mocking da API
- `test_summarize_text_api_call_format`: Verifica formato da chamada
- `test_generate_text_different_temperature`: Valida temperatura para funções criativas
- `test_translate_text_includes_target_language`: Confirma inclusão do idioma alvo

### Validação
- `test_empty_text_handling`: Comportamento com texto vazio
- `test_none_text_raises_error`: Tratamento de valores None
- `test_max_tokens_validation`: Validação de limites de tokens
- `test_model_parameter_validation`: Validação de modelos

### Tratamento de Erros
- `test_api_error_handling`: Erros gerais da API
- `test_rate_limit_error_handling`: Limite de taxa
- `test_connection_error_handling`: Problemas de conexão
- `test_invalid_response_structure`: Respostas malformadas

### Integridade Funcional
- `test_system_messages_consistency`: Consistência de mensagens do sistema
- `test_creative_functions_use_higher_temperature`: Temperatura para criatividade
- `test_analytical_functions_use_lower_temperature`: Temperatura para análise
- `test_response_content_stripping`: Limpeza de espaços em branco

## Benefícios

1. **Economia**: Não faz chamadas reais à API durante testes
2. **Rapidez**: Testes executam em segundos
3. **Confiabilidade**: Detecta quebras antes da produção
4. **Documentação**: Exemplifica como usar as funções
5. **Coverage**: Identifica código não testado

## Métricas Esperadas

- **Coverage**: >90% das linhas de código
- **Tempo de execução**: <30 segundos
- **Número de testes**: 20+ cenários diferentes

## Extensão dos Testes

Para adicionar novos testes:

1. **Para nova função**:
```python
@patch('app.openai_service.client.chat.completions.create')
def test_nova_funcao(self, mock_create):
    mock_create.return_value = self.mock_response
    result = nova_funcao("input")
    self.assertIsNotNone(result)
```

2. **Para novo tipo de erro**:
```python
@patch('app.openai_service.client.chat.completions.create')
def test_novo_erro(self, mock_create):
    mock_create.side_effect = NovoTipoErro("Mensagem")
    with self.assertRaises(NovoTipoErro):
        funcao_qualquer("input")
```

## Troubleshooting

**Erro de import**: Verifique se está no diretório correto com `openai_service.py`

**Erro de OpenAI**: Instale: `pip install openai`

**Erro de mock**: O unittest.mock faz parte do Python 3.3+

**Testes lentos**: Verifique se não está fazendo chamadas reais à API

---

*Estes testes foram criados seguindo as melhores práticas de TDD e garantem a qualidade e confiabilidade do código.*
