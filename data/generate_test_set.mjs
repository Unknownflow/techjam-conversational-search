import { randomInt } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

const CATALOG_PATH = "data/catalog.jsonl";
const PUBLIC_SET_PATH = "data/public_set.jsonl";
const OUTPUT_PATH = "data/test_set.jsonl";

const SCENARIO_COUNTS = {
  buying: 400,
  browsing: 400,
  intent_override: 150,
  boundary: 50,
};

const TAG_WEIGHTS = [
  ["fit", 163],
  ["material", 154],
  ["comfort", 144],
  ["style", 101],
  ["durability", 47],
  ["performance", 26],
  ["warmth", 18],
  ["weather", 12],
  ["general shopping", 1],
];

function parseJsonl(path) {
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function shuffle(values) {
  const result = [...values];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = randomInt(index + 1);
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
}

function weightedIndex(items) {
  const totalWeight = items.reduce((total, [, weight]) => total + weight, 0);
  let cursor = randomInt(totalWeight);
  for (let index = 0; index < items.length; index += 1) {
    cursor -= items[index][1];
    if (cursor < 0) return index;
  }
  return items.length - 1;
}

function profile() {
  const ratings = [
    [5.0, 134, "usually positive"],
    [4.0, 21, "mixed"],
    [3.0, 22, "critical"],
    [2.0, 9, "critical"],
    [1.0, 14, "critical"],
  ];
  const [averagePriorRating, , ratingStyle] =
    ratings[weightedIndex(ratings.map(([rating, weight]) => [rating, weight]))];
  const tagCount = [1, 2, 3, 4][
    weightedIndex([
      [1, 12],
      [2, 57],
      [3, 49],
      [4, 82],
    ])
  ];
  const tags = [];
  const available = [...TAG_WEIGHTS];
  while (tags.length < tagCount) {
    const index = weightedIndex(available);
    tags.push(available[index][0]);
    available.splice(index, 1);
  }
  return {
    average_prior_rating: averagePriorRating,
    preference_tags: tags,
    purchase_frequency: "3-4 prior purchases",
    rating_style: ratingStyle,
    summary: `Prior purchases emphasize ${tags.join(", ")}; ratings are ${ratingStyle}.`,
  };
}

const publicTargets = new Set(parseJsonl(PUBLIC_SET_PATH).map((sample) => String(sample.ground_truth.parent_asin)));
const uniqueCatalogTargets = new Map();
for (const product of parseJsonl(CATALOG_PATH)) {
  const parentAsin = String(product.parent_asin || "").trim();
  if (parentAsin && !publicTargets.has(parentAsin) && !uniqueCatalogTargets.has(parentAsin)) {
    uniqueCatalogTargets.set(parentAsin, product);
  }
}

const scenarios = shuffle(Object.entries(SCENARIO_COUNTS).flatMap(([scenario, count]) => Array(count).fill(scenario)));
const targets = shuffle([...uniqueCatalogTargets.keys()]).slice(0, scenarios.length);
if (targets.length !== scenarios.length) {
  throw new Error(`Expected ${scenarios.length} unique holdout targets; found ${targets.length}.`);
}

const samples = shuffle(
  targets.map((parentAsin, index) => {
    const scenarioType = scenarios[index];
    return {
      category_bucket: "clothing",
      difficulty_bucket: scenarioType === "buying" ? "easy" : scenarioType === "intent_override" ? "hard" : "medium",
      ground_truth: { parent_asin: parentAsin },
      scenario_type: scenarioType,
      user_profile: profile(),
    };
  }),
).map((sample, index) => ({
  ...sample,
  sample_id: `test_${String(index + 1).padStart(4, "0")}`,
}));

const targetIds = samples.map((sample) => sample.ground_truth.parent_asin);
const sampleIds = samples.map((sample) => sample.sample_id);
if (new Set(targetIds).size !== samples.length || new Set(sampleIds).size !== samples.length) {
  throw new Error("Generated set contains duplicate target or sample identifiers.");
}
for (const [scenario, count] of Object.entries(SCENARIO_COUNTS)) {
  if (samples.filter((sample) => sample.scenario_type === scenario).length !== count) {
    throw new Error(`Generated ${scenario} count does not match ${count}.`);
  }
}

writeFileSync(OUTPUT_PATH, `${samples.map((sample) => JSON.stringify(sample)).join("\n")}\n`, "utf8");
