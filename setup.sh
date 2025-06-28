#!/bin/bash

# Verifica se o script está sendo executado com source
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "⚠️  Para exportar as variáveis no shell atual, execute:"
    echo "   source ./setup.sh"
    echo "   ou"
    echo "   . ./setup.sh"
    echo ""
fi

# Verifica se o arquivo .env existe
if [ ! -f .env ]; then
    echo "Arquivo .env não encontrado no diretório atual"
    exit 1
fi

echo "Exportando variáveis de ambiente do arquivo .env..."

# Lê o arquivo .env e exporta cada variável
while IFS= read -r line || [ -n "$line" ]; do
    # Ignora linhas vazias e comentários
    if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
        # Remove espaços em branco no início e fim
        line=$(echo "$line" | xargs)
        
        # Verifica se a linha contém um '='
        if [[ "$line" =~ = ]]; then
            # Exporta a variável
            export "$line"
            echo "Exportada: $(echo "$line" | cut -d'=' -f1)"
        fi
    fi
done < .env

echo "Variáveis de ambiente exportadas com sucesso!"

# Verifica se uma variável específica foi exportada (para debug)
if [[ -n "$OPENAI_API_KEY" ]]; then
    echo "✅ OPENAI_API_KEY está disponível"
else
    echo "❌ OPENAI_API_KEY não encontrada"
fi