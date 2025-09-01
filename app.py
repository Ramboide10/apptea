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

# Dicionário (Mem. Ram) para armazenar resultados
resultados = OrderedDict()
MAX_RESULTADOS = 5000

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'encoder.pkl')
FEATURE_COLS_PATH = os.path.join(BASE_DIR, 'feature_cols.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'meu_modelo_TEA.keras')
THRESHOLD_PATH = os.path.join(BASE_DIR, 'threshold.pkl')

# Carrega modelo e pré-processadores
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_cols = pd.read_csv(FEATURE_COLS_PATH, header=None).iloc[:, 0].tolist()
model = load_model(MODEL_PATH)
t_o = joblib.load(THRESHOLD_PATH)


# Configura a URL base-------------------------------------+
APP_URL = os.environ.get('APP_URL', 'http://127.0.0.1:5000') 
#----------------------------------------------------------+
#
#
#-------------- Início de função: get_latest_form_data -------------+
#
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
        return latest_row, id_cliente
#
#-------------- Fim de função: get_latest_form_data ---------------+
#
# --------------------+ R O T A S +-----------------------------+
@app.route('/')   # ← Rota raiz do site
def home():
    return """
    <h1>Aplicativo TEA-Adoslecente</h1>
    <p>Servidor Flask está funcionando corretamente!</p>

    <a href="https://sites.google.com/view/profmat-csa-ufsj/home" 
    style="color: #337ab7; text-decoration: none;">
    ← Voltar à Página Principal do Aplicativo
    """
#---------------------------------------------------------------+
#
# Para acessar resultados da avaliação TEA
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Verificação do token de acesso básico
        token = request.args.get('token')
        id_requerido = request.args.get('ID')  # Novo parâmetro: ID do usuário
        
        
        # Verifica o token básico
        if token != TOKEN_ACESSO:
            return Response(
                """
                <div style="font-family: Arial; text-align: center; margin-top: 50px;">
                
                    <h2 style="color: #d9534f;">Acesso Restrito</h2>
                    
                    <a href="https://sites.google.com/view/profmat-csa-ufsj/home" 
                       style="color: #337ab7; text-decoration: none;">
                        ← Voltar à Página Principal do Aplicativo
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
                
                    <h2 style="color: #b38600;">Aplicativo TEA-Adoslecente</h2>

                    <!-- Botão para redirecionamento automático -->
                    <a href="{APP_URL}/predict?token=TEA12345&ID={id_cliente}"
                        style="display: inline-block; margin-top: 20px; padding: 10px 20px; 
                            background-color: #0099cc; color: white; text-decoration: none; 
                            border-radius: 5px; font-weight: bold;">
                            
                        Ver meu Resultado
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
                        ← Voltar à Página Principal do Aplicativo
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
            # Renomeia colunas para: Col1, Col2, ... Col17---------------------------------+
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

            #----------------------------------------------------------------------------------------------+
            #++++++++++++++++++++++++++++++++++++++++++++++ooooooooooooooooooooooooooooooooooooooooooooooooo
            # Processa a P R E D I Ç Ã O
            df_cliente_transformado = transform_new(df_cliente, scaler, encoder, feature_cols)
            resultado = float(model.predict(df_cliente_transformado.values.reshape(1, -1))[0][0])
            #-----------------------------------------------------------------------------------------------+
            
            # Aplicar threshold ajustado---+
            if resultado > t_o:
                classe_predita = 1
                decisao = "TEA"
            else:
                classe_predita = 0
                decisao = "Não TEA"
            #-------------------------------+
                
            # Classificação-----------------------------------------------------------+
            Delta = 0.5*t_o
            if t_o == 0.5:
                if 0 <= resultado < Delta:
                    classificacao = "Baixa"                 #classificação = nível
                    interpretacao = "improvável"
                elif Delta <= resultado < 2*Delta:
                    classificacao = "Leve"
                    interpretacao = "possível"
                elif 2*Delta <= resultado < 3*Delta:
                    classificacao = "Moderada"
                    interpretacao = "provável"
                elif 3*Delta <= resultado <= 1:
                    classificacao = "Alta"
                    interpretacao = "muito provável"
        
            elif t_o == 0.45:
                if 0 <= resultado < Delta:
                    classificacao = "Baixa"                 #classificação = nível
                    interpretacao = "improvável"
                elif Delta <= resultado < 2*Delta:
                    classificacao = "Sinal Inicial"
                    interpretacao = "recomenda-se observação"
                elif 2*Delta <= resultado < 0.55:
                    classificacao = "Leve"
                    interpretacao = "possível"
                elif 0.55 <= resultado < 0.775:
                    classificacao = "Moderada"
                    interpretacao = "provável"
                elif 0.775 <= resultado <= 1:
                    classificacao = "Alta"
                    interpretacao = "muito provável"
                    
            elif t_o == 0.4:
                if 0 <= resultado < Delta:
                    classificacao = "Baixa"                 #classificação = nível
                    interpretacao = "improvável"
                elif Delta <= resultado < 2*Delta:
                    classificacao = "Baixa a Leve"
                    interpretacao = "possibilidade não descartada"
                elif 2*Delta <= resultado < 3*Delta:
                    classificacao = "Leve"
                    interpretacao = "possível"
                elif 3*Delta <= resultado < 4*Delta:
                    classificacao = "Moderada"
                    interpretacao = "provável"
                elif 4*Delta <= resultado <= 1:
                    classificacao = "Alta"
                    interpretacao = "muito provável"      
            #--------------------------------------------------------------------------+
                
            # Armazena o resultado (Funciona em quanto o código está sendo executado)--+
            resultados[id_cliente] = {
                'probabilidade': resultado,
                'classificacao': classificacao,
                'interpretacao': interpretacao,   # <-- adicionado
                'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
            #---------------------------------------------------------------------------+
            
            # Limita o tamanho do dicionário de resultados
            if len(resultados) > MAX_RESULTADOS:
                resultados.popitem(last=False)
            
            resultado_data = resultados[id_cliente]

            # RESUMO: RESULTADOS PARA VISUALIZAÇÃO--------------------------------------------------+
            cdt = 1                                                                                 #
            if cdt == 1:                                                                            #
                print('')                                                                           #
                print(f'id_requerido={id_requerido}')                                               #
                print(f"Probabilidade predita: {resultado:.4f}")                                    #
                print(f"Classe predita (threshold {t_o}): {classe_predita} ({decisao})")            #
                print(f"Resultado: Probabilidade de TEA {resultado_data['classificacao'].lower()}") #
            #---------------------------------------------------------------------------------------+
            
        # Gera o HTML
        resultado_html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 20px auto;">
            <h2 style="color: #2c3e50;">Resultado da Avaliação</h2>
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
                <p><strong>ID:</strong> {id_cliente}</p>
                <hr>
                <p><strong>Resultado:</strong> Probabilidade de TEA <strong>{resultado_data['classificacao'].lower()}</strong></p>
                
                <p><strong>Interpretação do resultado:</strong></p>
                
                <p style="font-size: 0.9em;">Com base nas informações fornecidas, a probabilidade de 
                Transtorno do Espectro Autista aponta para uma ocorrência <strong>{resultado_data['interpretacao'].lower()}</strong>.</p>
                

                <h4 style="margin-top: 20px;">Faixas de classificação:</h4>
                   
        """
        if t_o ==0.5:  
            resultado_html += """
                <div style="
                display: inline-grid;
                grid-template-columns: 68px auto auto;
                column-gap: 20px;
                row-gap: 6px;
                font-size: 0.85rem;
                line-height: 1.3
                ">

                <div><strong>Baixa</strong></div>     <div>&lt; 0.25 </div>  <div>Improvável</div>
                <div><strong>Leve</strong></div>      <div>&lt; 0.5  </div>  <div>Possível</div>
                <div><strong>Moderada</strong></div>  <div>&lt; 0.75 </div>  <div>Provável</div>
                <div><strong>Alta</strong></div>      <div>&le; 1    </div>  <div>Muito provável</div>
                </div>
            """
        elif t_o ==0.45:
            resultado_html += """
                <div style="
                display: inline-grid;
                grid-template-columns: 88px auto auto;
                column-gap: 20px;
                row-gap: 6px;
                font-size: 0.85rem;
                line-height: 1.3
                ">

                <div><strong>Baixa</strong></div>             <div>&lt; 0.225</div>  <div>Improvável</div>
                <div><strong>Sinal inicial</strong></div>     <div>&lt; 0.45 </div>  <div>Recomenda-se observação</div>
                <div><strong>Leve</strong></div>              <div>&lt; 0.55 </div>  <div>Possível</div>
                <div><strong>Moderada</strong></div>          <div>&lt; 0.775</div>  <div>Provável</div>
                <div><strong>Alta</strong></div>              <div>&le; 1    </div>  <div>Muito provável</div>
                </div>
            """ 
        elif t_o ==0.4:
            resultado_html += """
                <div style="
                display: inline-grid;
                grid-template-columns: 88px auto auto;
                column-gap: 20px;
                row-gap: 6px;
                font-size: 0.85rem;
                line-height: 1.3
                ">

                <div><strong>Baixa</strong></div>             <div>&lt; 0.2 </div>  <div>Improvável</div>
                <div><strong>Baixa a leve</strong></div>      <div>&lt; 0.4 </div>  <div>Possibilidade não descartada</div>
                <div><strong>Leve</strong></div>              <div>&lt; 0.6 </div>  <div>Possível</div>
                <div><strong>Moderada</strong></div>          <div>&lt; 0.8 </div>  <div>Provável</div>
                <div><strong>Alta</strong></div>              <div>&le; 1   </div>  <div>Muito provável</div>
                </div>
            """                   
        
        resultado_html += f"""        
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
