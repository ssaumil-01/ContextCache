import { default as runTest, exactCounter, semanticCounter, novelCounter } from './load_test.js';

export const options = {
  vus: 100,
  duration: '2m',
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<300'],
  },
};

export { exactCounter, semanticCounter, novelCounter };
export default runTest;
