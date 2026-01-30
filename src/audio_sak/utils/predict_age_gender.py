#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import glob
import pandas as pd
import math
import logging
import sys

try:
    import audeer
    import audonnx
    import audinterface
except ImportError:
    audeer = None
    audonnx = None
    audinterface = None

logger = logging.getLogger(__name__)

# Configurações do modelo
MODEL_URL = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'
SAMPLING_RATE = 16000

def predict_age_gender(target_dir, output_dir, batch_size=1000, cache_root='cache', model_root='model_age_gender'):
    if any(x is None for x in [audeer, audonnx, audinterface]):
        logger.error("audeer, audonnx, or audinterface not installed.")
        return

    logger.info("--- Script de Predição de Idade e Gênero ---")

    # --- 2. Download e Extração do Modelo (se necessário) ---
    logger.info("[ETAPA 1/6] Verificando e configurando o modelo...")
    audeer.mkdir(cache_root)
    model_zip_path = os.path.join(cache_root, 'w2v2-age-gender.zip')

    if not os.path.exists(model_root):
        if not os.path.exists(model_zip_path):
            logger.info(f"Baixando o modelo de '{MODEL_URL}'...")
            audeer.download_url(MODEL_URL, model_zip_path, verbose=True)
        logger.info(f"Extraindo modelo para o diretório '{model_root}'...")
        audeer.extract_archive(model_zip_path, model_root, verbose=True)
    else:
        logger.info(f"Diretório do modelo '{model_root}' já existe. Etapa pulada.")

    # --- 3. Carregamento do Modelo e Interface de Processamento ---
    logger.info("[ETAPA 2/6] Carregando o modelo na memória...")
    model = audonnx.load(model_root)
    outputs = ['logits_age', 'logits_gender']
    interface = audinterface.Feature(
        model.labels(outputs),
        process_func=model,
        process_func_args={'outputs': outputs, 'concat': True},
        sampling_rate=SAMPLING_RATE,
        resample=True,
        verbose=True,
    )

    # --- 4. Encontrando TODOS os Arquivos de Áudio ---
    logger.info(f"[ETAPA 3/6] Procurando por arquivos .flac/.wav em '{target_dir}'...")
    # Support flac and wav
    all_file_paths = glob.glob(os.path.join(target_dir, '**', '*.flac'), recursive=True) + \
                     glob.glob(os.path.join(target_dir, '**', '*.wav'), recursive=True)

    if not all_file_paths:
        logger.warning(f"AVISO: Nenhum arquivo .flac/.wav foi encontrado no diretório especificado.")
        return

    total_files = len(all_file_paths)
    total_batches = math.ceil(total_files / batch_size)
    logger.info(f"Encontrados {total_files} arquivos. Serão processados em {total_batches} lotes de ~{batch_size} arquivos.")

    os.makedirs(output_dir, exist_ok=True)

    # --- 5. Processamento em Lotes ---
    for i in range(total_batches):
        batch_num = i + 1
        logger.info(f"\n--- Processando Lote {batch_num}/{total_batches} ---")

        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_paths = all_file_paths[start_index:end_index]

        if not batch_paths:
            continue

        # [ETAPA 4/6] Criando o índice para o lote atual
        logger.info(f"[ETAPA 4/6] Criando índice para {len(batch_paths)} arquivos...")
        file_index = pd.Index(batch_paths, name='file')

        # [ETAPA 5/6] Executando a Predição para o lote
        logger.info(f"[ETAPA 5/6] Iniciando a predição para o lote {batch_num}...")
        predictions_raw = interface.process_index(file_index, root='/', cache_root=cache_root)

        # [ETAPA 6/6] Pós-processamento e Geração do CSV para o lote
        logger.info(f"[ETAPA 6/6] Processando resultados e gerando o relatório CSV para o lote {batch_num}...")

        predicted_age = predictions_raw['age'] * 100
        gender_columns = ['female', 'male', 'child']
        predicted_gender = predictions_raw[gender_columns].idxmax(axis=1)

        results_df = pd.DataFrame({
            'filepath': predictions_raw.index.get_level_values('file'),
            'predicted_age': predicted_age.values,
            'predicted_gender': predicted_gender.values
        })

        output_filename = os.path.join(output_dir, f'age_gender_predictions_part_{batch_num}.csv')
        results_df.to_csv(output_filename, index=False)
        logger.info(f"Resultados do lote {batch_num} salvos em '{output_filename}'")

    logger.info(f"\n--- Processamento de todos os {total_batches} lotes concluído! ---")

def main():
    parser = argparse.ArgumentParser(description="Predict Age and Gender from Audio.")
    parser.add_argument('-i', '--input', required=True, help='Input directory')
    parser.add_argument('-o', '--output', default='output_predictions', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1000)
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    predict_age_gender(args.input, args.output, args.batch_size)

if __name__ == "__main__":
    main()

