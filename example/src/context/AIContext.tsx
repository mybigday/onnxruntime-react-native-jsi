import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from 'react';
import { pipeline, TextStreamer } from '@huggingface/transformers';
import PerformanceStats from 'react-native-performance-stats';
import { AudioContext } from 'react-native-audio-api';

export type AITask =
  | 'text-generation'
  | 'automatic-speech-recognition'
  | 'text-to-audio';
export type DType =
  | 'auto'
  | 'fp32'
  | 'fp16'
  | 'q8'
  | 'q4'
  | 'q4f16'
  | 'int8'
  | 'uint8';

export type ChatMessage = {
  role: 'user' | 'assistant' | 'system';
  content: string;
};

type LoadOptions = {
  dtype?: DType; // transformers.js quantization dtype
};

type RunChatOptions = {
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
};

type TTSResult = { audio?: Float32Array; sampling_rate?: number } | undefined;

type ASRResult = { text?: string } | undefined;

type AIContextType = {
  task: AITask;
  setTask: (task: AITask) => void;
  modelId: string;
  setModelId: (id: string) => void;
  dtype: DType;
  setDtype: (dtype: DType) => void;
  isLoading: boolean;
  isLoaded: boolean;
  error?: unknown;
  loadPipeline: () => Promise<void>;
  unloadPipeline: () => void;
  // Text-generation (chat)
  runChat: (
    messages: ChatMessage[] | string,
    onToken?: (text: string) => void,
    options?: RunChatOptions
  ) => Promise<string>;
  // ASR
  runASRFromUrl: (audioUrl: string) => Promise<ASRResult>;
  runASRFromFloat32: (pcm: Float32Array) => Promise<ASRResult>;
  // TTS
  runTTS: (text: string) => Promise<TTSResult>;
};

const defaultModels: Record<AITask, string> = {
  'text-generation': 'onnx-community/Qwen3-0.6B-ONNX',
  'automatic-speech-recognition': 'onnx-community/whisper-small',
  'text-to-audio': 'Xenova/mms-tts-eng',
};

const AIContext = createContext<AIContextType | undefined>(undefined);

async function watchMemoryUsage<T>(fn: () => Promise<T>): Promise<[T, number]> {
  let peakMem = -1;
  PerformanceStats.start();
  const listener = PerformanceStats.addListener((stats) => {
    peakMem = Math.max(peakMem, stats.usedRam);
  });
  try {
    return [await fn(), peakMem];
  } finally {
    listener.remove();
    PerformanceStats.stop();
  }
}

export const AIProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [task, setTask] = useState<AITask>('text-generation');
  const [modelId, setModelId] = useState<string>(
    defaultModels['text-generation']
  );
  const [dtype, setDtype] = useState<DType>('auto');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<unknown>(undefined);

  const pipeRef = useRef<any | null>(null);

  const loadPipeline = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(undefined);
      const opts: LoadOptions = { dtype };
      await pipeRef.current?.dispose();
      console.log('modelId:', modelId);
      console.log('dtype:', dtype);
      const [pipe, peakMem] = await watchMemoryUsage(() =>
        pipeline(task as any, modelId, opts as any)
      );
      pipeRef.current = pipe;
      console.log('memory usage after load:', peakMem);
      setIsLoaded(true);
    } catch (e) {
      setError(e);
      throw e;
    } finally {
      setIsLoading(false);
    }
  }, [task, modelId, dtype]);

  const unloadPipeline = useCallback(async () => {
    await pipeRef.current?.dispose();
    pipeRef.current = null;
    setIsLoaded(false);
  }, []);

  const runChat = useCallback(
    async (
      messages: ChatMessage[] | string,
      onToken?: (text: string) => void,
      options?: RunChatOptions
    ): Promise<string> => {
      if (!pipeRef.current) throw new Error('Pipeline is not loaded');
      let full = '';
      let tokens = 0;
      const streamer = new TextStreamer(pipeRef.current.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (text: string) => {
          full += text;
          tokens += 1;
          onToken?.(text);
        },
      });
      const input = pipeRef.current.tokenizer.apply_chat_template(
        Array.isArray(messages)
          ? messages
          : [{ role: 'user', content: messages }],
        {
          add_generation_prompt: true,
          tokenize: false,
        }
      );
      tokens = pipeRef.current.tokenizer.encode(input).length;
      const start = performance.now();
      const [_, peakMem] = await watchMemoryUsage(() =>
        pipeRef.current(input, {
          streamer,
          max_new_tokens: 512,
          ...(options || {}),
          do_sample: true,
        })
      );
      const end = performance.now();
      console.log('memory usage:', peakMem);
      console.log(`time taken: ${end - start}ms`);
      console.log(
        `tokens: ${tokens}, tokens/s: ${(tokens / (end - start)) * 1000}`
      );
      return full;
    },
    []
  );

  const runASRFromUrl = useCallback(
    async (audioUrl: string): Promise<ASRResult> => {
      if (!pipeRef.current) throw new Error('Pipeline is not loaded');
      const audioContext = new AudioContext({
        sampleRate: 16000,
      });
      const res = await fetch(audioUrl);
      const buf = await res.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(buf);
      const pcm = audioBuffer.getChannelData(0);
      const out = await pipeRef.current(pcm);
      return out as ASRResult;
    },
    []
  );

  const runASRFromFloat32 = useCallback(
    async (pcm: Float32Array): Promise<ASRResult> => {
      if (!pipeRef.current) throw new Error('Pipeline is not loaded');
      const out = await pipeRef.current(pcm);
      return out as ASRResult;
    },
    []
  );

  const runTTS = useCallback(async (text: string): Promise<TTSResult> => {
    if (!pipeRef.current) throw new Error('Pipeline is not loaded');
    const out = await pipeRef.current(text);
    return out as TTSResult;
  }, []);

  const value = useMemo<AIContextType>(
    () => ({
      task,
      setTask: (t: AITask) => {
        setTask(t);
        setModelId(defaultModels[t]);
      },
      modelId,
      setModelId,
      dtype,
      setDtype,
      isLoading,
      isLoaded,
      error,
      loadPipeline,
      unloadPipeline,
      runChat,
      runASRFromUrl,
      runASRFromFloat32,
      runTTS,
    }),
    [
      task,
      modelId,
      dtype,
      isLoading,
      isLoaded,
      error,
      loadPipeline,
      unloadPipeline,
      runChat,
      runASRFromUrl,
      runASRFromFloat32,
      runTTS,
    ]
  );

  return <AIContext.Provider value={value}>{children}</AIContext.Provider>;
};

export const useAI = (): AIContextType => {
  const ctx = useContext(AIContext);
  if (!ctx) throw new Error('useAI must be used within AIProvider');
  return ctx;
};
