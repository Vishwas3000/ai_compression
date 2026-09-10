#!/usr/bin/env node

import { existsSync, readFileSync } from "node:fs";
import { basename, dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const env = join(here, ".env");
if (existsSync(env)) process.loadEnvFile(env);

const args = process.argv.slice(2);
const dryRun = args.includes("--dry-run");
const argument = args.find((value) => !value.startsWith("-")) || "jpeg-ai-on-apple-silicon";
const filename = extname(argument) ? basename(argument) : `${basename(argument)}.md`;
const path = join(here, filename);
if (!existsSync(path)) throw new Error(`No article at ${path}`);

const raw = readFileSync(path, "utf8");
const frontMatter = raw.match(/^---\n([\s\S]*?)\n---\n/);
if (!frontMatter) throw new Error(`${filename} has no front matter`);
const meta = Object.fromEntries(frontMatter[1].split("\n").map((line) => {
  const match = line.match(/^(\w+):\s*(.*)$/);
  return match ? [match[1], match[2].replace(/^["']|["']$/g, "").trim()] : [];
}).filter((entry) => entry.length));
const body = raw.slice(frontMatter[0].length).trim();
const tags = (meta.tags || "").split(",").map((tag) => tag.trim()).filter(Boolean);

if (!meta.title || !meta.description || !meta.cover_image) {
  throw new Error("title, description, and cover_image are required");
}
if (meta.published !== "false") throw new Error("published must remain false; publish manually on DEV");
if (!tags.length || tags.length > 4) throw new Error("DEV requires one to four tags");

console.log(`${meta.title}\n  tags   ${tags.join(", ")}\n  cover  ${meta.cover_image}`);
if (dryRun) {
  console.log(`  body   ${body.split(/\s+/).length} words\n\nDry run passed; no network request was made.`);
  process.exit(0);
}

const key = process.env.DEVTO_API_KEY;
if (!key) throw new Error("DEVTO_API_KEY is missing; copy blog/.env.example to blog/.env");
const cover = await fetch(meta.cover_image, { method: "HEAD", redirect: "follow" });
if (!cover.ok) throw new Error(`Cover image is not public yet (${cover.status}); push the assets first`);

const headers = { "api-key": key, "content-type": "application/json" };
const listing = await fetch("https://dev.to/api/articles/me/all?per_page=100", { headers });
if (!listing.ok) throw new Error(`DEV returned ${listing.status} while listing articles`);
const existing = (await listing.json()).find((article) => article.title === meta.title);
if (existing?.published) throw new Error("The matching DEV article is already public; edit it manually");

const response = await fetch(
  existing ? `https://dev.to/api/articles/${existing.id}` : "https://dev.to/api/articles",
  {
    method: existing ? "PUT" : "POST",
    headers,
    body: JSON.stringify({
      article: {
        title: meta.title,
        description: meta.description,
        body_markdown: body,
        main_image: meta.cover_image,
        tags,
        published: false,
      },
    }),
  },
);
const result = await response.json().catch(() => ({}));
if (!response.ok) throw new Error(`DEV ${response.status}: ${result.error || "request failed"}`);
console.log(`\n${existing ? "Updated" : "Created"} draft: ${result.url || "https://dev.to/dashboard"}/edit`);
