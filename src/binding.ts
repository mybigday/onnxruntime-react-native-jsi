import mod from './native';

if (typeof globalThis.OrtApi === 'undefined') {
  mod.install();
}

export const OrtApi = globalThis.OrtApi;
