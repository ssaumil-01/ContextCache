import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter } from 'k6/metrics';

// -------------------------------------------------------------
// k6 Load Test Configuration
// -------------------------------------------------------------
export const options = {
  stages: [
    { duration: '30s', target: 10 },  // Stage 1: Warmup & baseline (10 VUs)
    { duration: '30s', target: 50 },  // Stage 2: Scale ramp-up (50 VUs)
    { duration: '30s', target: 100 }, // Stage 3: Normal capacity (100 VUs)
    { duration: '30s', target: 500 }, // Stage 4: Stress capacity (500 VUs)
    { duration: '30s', target: 0 },   // Cool-down
  ],
  thresholds: {
    http_req_failed: ['rate<0.01'],   // Failures must be less than 1%
    http_req_duration: ['p(95)<300'], // 95% of request latencies must be under 300ms
  },
};

// Custom metrics to track query distribution in the workload
export const exactCounter = new Counter('exact_query_count');
export const semanticCounter = new Counter('semantic_query_count');
export const novelCounter = new Counter('novel_query_count');

// Load query workload dataset
const workload = JSON.parse(open('./benchmark_workload.json'));

const tenants = ["tenant_a", "tenant_b", "tenant_c", null];

export default function () {
  const item = workload[Math.floor(Math.random() * workload.length)];
  const query = item.query;
  const type = item.type;

  // Track counts for workload distribution verification
  if (type === 'exact') {
    exactCounter.add(1);
  } else if (type === 'semantic') {
    semanticCounter.add(1);
  } else if (type === 'novel') {
    novelCounter.add(1);
  }

  const tenant_id = tenants[Math.floor(Math.random() * tenants.length)];

  const payload = JSON.stringify({
    query: query,
    tenant_id: tenant_id,
    scope: tenant_id ? "tenant" : "global",
    doc_ids: ["doc_id_1", "doc_id_2"]
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post('http://localhost:8000/query', payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has cache_hit property': (r) => {
      try { return r.json().hasOwnProperty('cache_hit'); } catch (e) { return false; }
    },
    'has source property': (r) => {
      try { return r.json().hasOwnProperty('source'); } catch (e) { return false; }
    },
  });

  // Small pacing delay to simulate realistic staggered human typing queries
  sleep(Math.random() * 0.1 + 0.05); // Sleep between 50ms and 150ms
}

