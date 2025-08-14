import { useEffect, useState, useRef, useCallback } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import {
  InferenceSession as JSIInferenceSession,
  Tensor as OrtTensor,
  // @ts-ignore
} from '@force/onnxruntime-react-native-jsi';
// @ts-ignore
import { InferenceSession as OldInferenceSession } from '@force/onnxruntime-react-native';
import type { InferenceSession } from 'onnxruntime-common';
import { AutoTokenizer, Tensor } from '@huggingface/transformers';
import * as RNFS from '@dr.pogodin/react-native-fs';
import PerformanceStats from 'react-native-performance-stats';
import bytes from 'bytes';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 20,
    marginBottom: 10,
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 10,
    marginTop: 20,
    fontWeight: 'bold',
  },
  result: {
    fontSize: 16,
  },
  spacer: {
    height: 10,
  },
});

const runCount = 20;

type Result = {
  total: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  peakMem: number;
};

const EMPTY_RESULT: Result = {
  total: 0,
  mean: 0,
  std: 0,
  min: 0,
  max: 0,
  peakMem: 0,
};

const MODEL_ID = 'onnx-community/bert-large-cased-ONNX';

const BENCHMARK_TEXT =
  `Very long text to test the performance of the inference session.`.repeat(32);

const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/model_q4.onnx`;

type BenchmarkReport = {
  times: number[];
  peakMem: number;
};

const benchmark = async (
  fn: () => Promise<void>,
  signal?: AbortSignal
): Promise<BenchmarkReport> => {
  let peakMem = -1;
  PerformanceStats.start();
  const listener = PerformanceStats.addListener((stats) => {
    peakMem = Math.max(peakMem, stats.usedRam);
  });
  try {
    const times = [];
    for (let i = 0; i < runCount && !signal?.aborted; i++) {
      const start = performance.now();
      await fn();
      const end = performance.now();
      times.push(end - start);
    }
    return { times, peakMem };
  } finally {
    listener.remove();
    PerformanceStats.stop();
  }
};

const calculateResult = (report: BenchmarkReport) => {
  const { times, peakMem } = report;
  const total = times.reduce((acc, time) => acc + time, 0);
  const mean = total / times.length;
  const std = Math.sqrt(
    times.reduce((acc, time) => acc + Math.pow(time - mean, 2), 0) /
      times.length
  );
  const min = Math.min(...times);
  const max = Math.max(...times);
  return { total, mean, std, min, max, peakMem };
};

const formatTime = (time: number) => {
  if (time < 1000) {
    return `${time.toFixed(2)} ms`;
  } else {
    return `${(time / 1000).toFixed(2)} s`;
  }
};

const options = {
  // enableCpuMemArena: true,
  // enableMemPattern: true,
  graphOptimizationLevel: 'all',
  interOpNumThreads: 4,
  freeDimensionOverrides: {
    batch_size: 1,
  },
};

export default function PerformanceTestScreen() {
  const [results, setResults] = useState<{
    jsi: Result;
    jsiPreAlloc: Result;
    old: Result;
  }>({
    jsi: EMPTY_RESULT,
    jsiPreAlloc: EMPTY_RESULT,
    old: EMPTY_RESULT,
  });
  const abortRef = useRef<AbortController | null>(null);
  const sessionRef = useRef<InferenceSession | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const runTest = useCallback(async () => {
    try {
      setIsRunning(true);

      const cachePath = RNFS.CachesDirectoryPath + '/bert-model.onnx';
      if (!(await RNFS.exists(cachePath))) {
        console.log('Downloading model...');
        const { promise } = RNFS.downloadFile({
          fromUrl: MODEL_URL,
          toFile: cachePath,
        });
        await promise;
        console.log('Model downloaded');
      }

      console.log('Tokenizing...');
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
      let input = await tokenizer(BENCHMARK_TEXT, {
        return_tensors: true,
      });
      input = Object.fromEntries(
        Object.entries(input as Record<string, Tensor>).map(([key, value]) => [
          key,
          value.ort_tensor,
        ])
      );
      console.log('Tokenized', input);

      const inputLength = (input.input_ids as Tensor).dims[1] as number;

      if (abortRef.current?.signal.aborted) return;

      let report: BenchmarkReport;

      // JSI InferenceSession
      {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        console.log('Benchmarking JSI InferenceSession...');
        const session = await JSIInferenceSession.create(cachePath, options);
        sessionRef.current = session;
        report = await benchmark(async () => {
          await session.run(input);
        }, abortRef.current?.signal);
        await session.release();
        sessionRef.current = null;
        const result = calculateResult(report);
        console.log('Benchmarking JSI InferenceSession done');

        setResults((prev) => ({
          ...prev,
          jsi: result,
        }));

        if (abortRef.current?.signal.aborted) return;
      }

      // JSI InferenceSession (Pre-allocated)
      {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        console.log('Benchmarking JSI InferenceSession (Pre-allocated)...');
        const session = await JSIInferenceSession.create(cachePath, options);
        sessionRef.current = session;
        const fetches = {
          logits: new OrtTensor(
            'float32',
            new Float32Array(inputLength * 30522),
            [1, inputLength, 30522]
          ),
        };
        report = await benchmark(async () => {
          await session.run(input, fetches);
        }, abortRef.current?.signal);
        await session.release();
        fetches.logits.dispose();
        sessionRef.current = null;
        const result = calculateResult(report);
        console.log('Benchmarking JSI InferenceSession (Pre-allocated) done');

        setResults((prev) => ({
          ...prev,
          jsiPreAlloc: result,
        }));

        if (abortRef.current?.signal.aborted) return;
      }

      // Old InferenceSession
      {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        console.log('Benchmarking Old InferenceSession...');
        const session = await OldInferenceSession.create(cachePath, options);
        sessionRef.current = session;
        report = await benchmark(async () => {
          await session.run(input);
        }, abortRef.current?.signal);
        await session.release();
        sessionRef.current = null;
        const result = calculateResult(report);
        console.log('Benchmarking Old InferenceSession done');

        setResults((prev) => ({
          ...prev,
          old: result,
        }));
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsRunning(false);
    }
  }, []);

  useEffect(() => {
    abortRef.current = new AbortController();

    return () => {
      abortRef.current?.abort();
      if (sessionRef.current) {
        sessionRef.current.release();
      }
    };
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Benchmarking with</Text>
      <Text style={styles.title}>{MODEL_ID}</Text>
      <View style={styles.spacer} />
      <Button title="Run Test" onPress={runTest} disabled={isRunning} />
      <Text style={styles.subtitle}>JSI InferenceSession</Text>
      <Text style={styles.result}>Total: {formatTime(results.jsi.total)}</Text>
      <Text style={styles.result}>Mean: {formatTime(results.jsi.mean)}</Text>
      <Text style={styles.result}>Std: {formatTime(results.jsi.std)}</Text>
      <Text style={styles.result}>Min: {formatTime(results.jsi.min)}</Text>
      <Text style={styles.result}>Max: {formatTime(results.jsi.max)}</Text>
      <Text style={styles.result}>
        Peak Memory Usage: {bytes(results.jsi.peakMem)}
      </Text>
      <Text style={styles.subtitle}>JSI InferenceSession (Pre-allocated)</Text>
      <Text style={styles.result}>
        Total: {formatTime(results.jsiPreAlloc.total)}
      </Text>
      <Text style={styles.result}>
        Mean: {formatTime(results.jsiPreAlloc.mean)}
      </Text>
      <Text style={styles.result}>
        Std: {formatTime(results.jsiPreAlloc.std)}
      </Text>
      <Text style={styles.result}>
        Min: {formatTime(results.jsiPreAlloc.min)}
      </Text>
      <Text style={styles.result}>
        Max: {formatTime(results.jsiPreAlloc.max)}
      </Text>
      <Text style={styles.result}>
        Peak Memory Usage: {bytes(results.jsiPreAlloc.peakMem)}
      </Text>
      <Text style={styles.subtitle}>Old InferenceSession</Text>
      <Text style={styles.result}>Total: {formatTime(results.old.total)}</Text>
      <Text style={styles.result}>Mean: {formatTime(results.old.mean)}</Text>
      <Text style={styles.result}>Std: {formatTime(results.old.std)}</Text>
      <Text style={styles.result}>Min: {formatTime(results.old.min)}</Text>
      <Text style={styles.result}>Max: {formatTime(results.old.max)}</Text>
      <Text style={styles.result}>
        Peak Memory Usage: {bytes(results.old.peakMem)}
      </Text>
    </View>
  );
}
