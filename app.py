# -*- coding: utf-8 -*-
from flask import Flask, jsonify, Response, request, abort
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
from collections import OrderedDict
import secrets

#------------------- DEFINIÇÃO DE FUNÇÕES ----------------------+
def transform_new(df_new, scaler, encoder, feature_cols):
    df = df_new.copy()
    yes_no_cols = ['Col01','Col02','Col03','Col04','Col05','Col06','Col07','Col08','Col09','Col10','Col14','Col16']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Sim': 1, 'Não': 0}).astype('int8')
    df.replace("?", np.nan, inplace=True)
    if 'Col12' in df.columns:
        df['Col12'] = df['Col12'].map({'Masculino': 0, 'Feminino': 1}).astype('int8')
    if 'Col11' in df.columns:
        df[['Col11']] = scaler.transform(df[['Col11']].astype(float))
    if df.shape[1] >= 10 and 'Col18' not in df.columns:
        df['Col18'] = df.iloc[:, :10].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    categorical_cols = ['Col13', 'Col15', 'Col17']
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].fillna('missing')
    present_cats = [c for c in categorical_cols if c in df.columns]
    if present_cats:
        encoded = encoder.transform(df[present_cats])
        feature_names = encoder.get_feature_names_out(present_cats)
        df = df.drop(columns=present_cats).reset_index(drop=True)
        df = pd.concat([df, pd.DataFrame(encoded, columns=feature_names)], axis=1)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df
#---------------- FIM DE DEFINIÇÃO DE FUNÇÕES ----------------+

app = Flask(__name__)

# Configuração de segurança
TOKEN_ACESSO = 'TEA12345'  # Token fixo para acesso básico
print('TOKEN_ACESSO=', TOKEN_ACESSO)

# Dicionário para armazenar resultados
resultados = OrderedDict()
MAX_RESULTADOS = 1000

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'encoder.pkl')
FEATURE_COLS_PATH = os.path.join(BASE_DIR, 'feature_cols.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'meu_modelo_TEA.h5')

# Carrega modelo e pré-processadores
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_cols = pd.read_csv(FEATURE_COLS_PATH, header=None).iloc[:, 0].tolist()
model = load_model(MODEL_PATH)


# Configura a URL base-------------------------------------+
APP_URL = os.environ.get('APP_URL', 'http://127.0.0.1:5000') 
#----------------------------------------------------------+


def get_latest_form_data(id_requerido=None):
    SHEET_ID = "19vZS3gvIQB_rbcixEgTy1rbkvkoeg1ywair4Ags7Rdk"
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv"
    df = pd.read_csv(url)

    if df.columns[0] == "Carimbo de data/hora":
        df.rename(columns={"Carimbo de data/hora": "ID"}, inplace=True)

    df.set_index("ID", inplace=True)
    
    if id_requerido:
        # Busca específica por um ID
        if id_requerido in df.index:
            return df.loc[[id_requerido]], id_requerido
        else:
            return None, None
    else:
        # Retorna o último registro (compatibilidade)
        latest_row = df.tail(1)
        id_cliente = latest_row.index[0]
        print('id_cliente=', id_cliente)
        return latest_row, id_cliente

# ---+ R O T A S +----+
@app.route('/')
def home():
    return """
    <h1>API de Predição TEA</h1>
    <p>Servidor Flask está funcionando corretamente!</p>
    <p>Use a rota <code>/predict</code> para acessar o modelo.</p>
    """
#
@app.route('/predict', methods=['GET'])
#
def predict():
    try:
        # Verificação do token de acesso básico
        token = request.args.get('token')
        id_requerido = request.args.get('ID')  # Novo parâmetro: ID do usuário
        
        print(f'token={token}, id_requerido={id_requerido}')
        
        # Verifica o token básico
        if token != TOKEN_ACESSO:
            return Response(
                """
                <div style="font-family: Arial; text-align: center; margin-top: 50px;">
                
                    <h2 style="color: #d9534f;">Acesso Restrito</h2>
                    
                    <p>Você precisa acessar através do link fornecido após o formulário.</p>
                    <a href="https://sites.google.com/view/profmat-csa-ufsj/home" 
                       style="color: #337ab7; text-decoration: none;">
                        ← Voltar ao site principal
                    </a>
                </div>
                """,
                mimetype='text/html',
                status=403
            )

        #--------------------------------------------------------------------------------+
        # Se não tem ID, mostra mensagem para preencher o formulário
        if not id_requerido:
            # Primeiro obtemos o ID do cliente (timestamp da última submissão)
            _, id_cliente = get_latest_form_data()
            return Response(
                f"""
                <div style="font-family: Arial; text-align: center; margin-top: 50px;">
                
                    <h2 style="color: #d9534f;">Aplicativo TEA-Adoslecente</h2>

                    <!-- Botão para redirecionamento automático -->
                    <a href="{APP_URL}/predict?token=TEA12345&ID={id_cliente}"
                        style="display: inline-block; margin-top: 20px; padding: 10px 20px; 
                            background-color: #4CAF50; color: white; text-decoration: none; 
                            border-radius: 5px; font-weight: bold;">
                        Ver Meu Resultado
                    </a>
            
                </div>
                """,
                mimetype='text/html',
                status=403
                )
        #--------------------------------------------------------------------------------+
        # Busca os dados específicos para o ID fornecido
        df_cliente, id_cliente = get_latest_form_data(id_requerido)
        
        # Verifica se encontrou o ID
        if df_cliente is None or id_cliente is None:
            return Response(
                """
                <div style="font-family: Arial; text-align: center; margin-top: 50px;">
                
                    <h2 style="color: #d9534f;">Registro Não Encontrado</h2>
                    
                    <p>Não foi encontrado um formulário com o ID fornecido.</p>
                    <p>Por favor, preencha o formulário primeiro para obter seu link personalizado.</p>
                    <a href="https://sites.google.com/view/profmat-csa-ufsj/home" 
                       style="color: #337ab7; text-decoration: none;">
                        ← Voltar ao site principal
                    </a>
                </div>
                """,
                mimetype='text/html',
                status=404
            )
        
        # Verifica se já temos um resultado armazenado para este ID
        if id_cliente in resultados:
            resultado_data = resultados[id_cliente]
        else:
            # Renomeia colunas se necessário---------------------------------+
            if df_cliente.columns.str.startswith(('1. ', '2. ', '3. ')).any():
                novo_nome_colunas = {
                    col: f"Col{int(col.split('.')[0]):02d}" 
                    for col in df_cliente.columns
                    if col.split('.')[0].isdigit()
                }
                df_cliente = df_cliente.rename(columns=novo_nome_colunas)

                
            #**********************************************************************************+
            # (Opcional) Exporta para Excel com colunas renomeadas
            ice = 0
            if ice == 1:
                output_filename = f"dados_cliente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                output_path = os.path.join(BASE_DIR, output_filename)
                df_cliente.to_excel(output_path, index=False, engine='openpyxl')
                print(f"Dados exportados para: {output_path}")
            #**********************************************************************************+

#---------------------------------------------+
            # Processa a P R E D I Ç Ã O
            df_cliente_transformado = transform_new(df_cliente, scaler, encoder, feature_cols)
            resultado = float(model.predict(df_cliente_transformado.values.reshape(1, -1))[0][0])
#---------------------------------------------+
            
            # Classificação
            if 0 <= resultado < 0.25:
                classificacao = "Baixo"
            elif 0.25 <= resultado < 0.5:
                classificacao = "Leve"
            elif 0.5 <= resultado < 0.75:
                classificacao = "Moderado"
            elif 0.75 <= resultado <= 1:
                classificacao = "Alto"
            else:
                classificacao = "Indefinida"

            # Armazena o resultado
            resultados[id_cliente] = {
                'probabilidade': resultado,
                'classificacao': classificacao,
                'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
            
            # Limita o tamanho do dicionário de resultados
            if len(resultados) > MAX_RESULTADOS:
                resultados.popitem(last=False)
            
            resultado_data = resultados[id_cliente]

        # Gera o HTML
        resultado_html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 20px auto;">
            <h2 style="color: #2c3e50;">Resultado da Avaliação</h2>
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
                <p><strong>ID:</strong> {id_cliente}</p>
                <hr>
                <p><strong>Probabilidade de TEA:</strong> {resultado_data['probabilidade']:.3f}</p>
                
                <p><strong>Interpretação do resultado:</strong> {resultado_data['classificacao']}</p>
                
                <p style="font-size: 0.9em;">Com base nas informações fornecidas, a probabilidade de 
                Transtorno do Espectro Autista está em um nível <strong>{resultado_data['classificacao'].lower()}</strong>.</p>

                
                <h4 style="margin-top: 20px;">Faixas de classificação:</h4>
                <ul>
                    <li><strong>Baixo</strong> (0-0.24) - Improvável</li>
                    <li><strong>Leve</strong> (0.25-0.49) - Possível</li>
                    <li><strong>Moderado</strong> (0.5-0.74) - Provável</li>
                    <li><strong>Alto</strong> (0.75-1) - Altamente Sugestivo</li>
                </ul>
                
                <hr>
                
                <p style="font-size: 0.7em;">
                    O link a seguir é pessoal e intransferível. Guarde-o com segurança:<br><br>
                    <a href="{APP_URL}/predict?token=TEA12345&ID={id_cliente}" style="color: #336699;">
                    {APP_URL}/predict?token=TEA12345&ID={id_cliente}
                    </a>
                </p>                
                <p style="font-size: 0.6em;">
                    Resultado gerado em: {resultado_data['timestamp']}
                </p>
            </div>
        </div>
        """
        
        return Response(resultado_html, mimetype='text/html')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
