export function parseInsightResponse(json, video) {
  const data = json?.data || json;
  if (!data) return { ok: false, reason: 'empty response' };

  const statusFlag = data?.status ?? json?.status;
  const dataQualityIssues = [];
  if (statusFlag === 2) dataQualityIssues.push('insufficient_data');

  const videoInfo =
    data?.video_info || data?.aweme_info || findFirstByKey(data, 'video_info') || {};
  const stats = videoInfo?.statistics || {};

  const retention = findInsight(data, 'video_retention_rate_realtime');
  const perDuration = findInsight(data, 'video_per_duration_realtime');
  const finishRate = findInsight(data, 'video_finish_rate_realtime');
  const trafficSource = findInsight(data, 'video_traffic_source_percent_realtime');
  const newFollowers = findInsight(data, 'video_new_followers');
  const totalViews = findInsight(data, 'realtime_total_video_views');

  const ecr = readRetentionAt(retention, '5000');
  if (ecr == null && statusFlag !== 2) dataQualityIssues.push('missing_ecr');

  const avgWatchTimeS = readNumericValue(perDuration);
  const durationMs = videoInfo?.video?.duration ?? videoInfo?.duration ?? video.duration_ms ?? null;
  const nawp = avgWatchTimeS != null && durationMs ? avgWatchTimeS / (durationMs / 1000) : null;

  const traffic = readTrafficSources(trafficSource);
  const createTs = videoInfo?.create_time ?? video.create_time;

  const row = {
    video_id: video.aweme_id,
    post_date: formatUnixDate(createTs),
    post_time: formatUnixTime(createTs),
    caption: videoInfo?.desc ?? video.desc ?? '',
    duration_ms: durationMs ?? video.duration_ms ?? '',
    views: readNumericValue(totalViews) ?? stats.play_count ?? '',
    likes: stats.digg_count ?? video.digg_count ?? '',
    comments: stats.comment_count ?? video.comment_count ?? '',
    shares: stats.share_count ?? video.share_count ?? '',
    ECR: ecr ?? '',
    avg_watch_time_s: avgWatchTimeS ?? '',
    NAWP: nawp != null ? roundTo(nawp, 6) : '',
    watched_full_pct: readNumericValue(finishRate) ?? '',
    traffic_foryou_pct: traffic.foryou ?? '',
    traffic_follow_pct: traffic.follow ?? '',
    traffic_profile_pct: traffic.profile ?? '',
    traffic_search_pct: traffic.search ?? '',
    new_followers: readNumericValue(newFollowers) ?? '',
    creator_uid: videoInfo?.author?.uid ?? '',
    creator_handle: videoInfo?.author?.unique_id ?? '',
    follower_count: '',
    account_created_date: '',
    data_quality: dataQualityIssues.join('|')
  };
  return { ok: true, row };
}

function findInsight(data, insighType) {
  if (!data || typeof data !== 'object') return null;
  if (Object.prototype.hasOwnProperty.call(data, insighType)) return data[insighType];
  const stack = [data];
  while (stack.length) {
    const node = stack.pop();
    if (!node || typeof node !== 'object') continue;
    if (Array.isArray(node)) {
      for (const item of node) {
        if (item && typeof item === 'object') {
          if (item.insigh_type === insighType || item.insight_type === insighType) return item;
          stack.push(item);
        }
      }
    } else {
      if (Object.prototype.hasOwnProperty.call(node, insighType)) return node[insighType];
      for (const key of Object.keys(node)) {
        const v = node[key];
        if (v && typeof v === 'object') stack.push(v);
      }
    }
  }
  return null;
}

function findFirstByKey(obj, key) {
  if (!obj || typeof obj !== 'object') return null;
  const stack = [obj];
  while (stack.length) {
    const node = stack.pop();
    if (!node || typeof node !== 'object') continue;
    if (Object.prototype.hasOwnProperty.call(node, key)) return node[key];
    if (Array.isArray(node)) for (const v of node) stack.push(v);
    else for (const k of Object.keys(node)) stack.push(node[k]);
  }
  return null;
}

function readNumericValue(node) {
  if (node == null) return null;
  const v = node?.value;
  if (v == null) return null;
  if (typeof v === 'object' && 'value' in v) {
    const n = Number(v.value);
    return Number.isFinite(n) ? n : null;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function readRetentionAt(node, timestamp) {
  const list = node?.value?.list ?? node?.value?.value?.list ?? node?.list;
  if (!Array.isArray(list)) return null;
  const target = String(timestamp);
  const entry = list.find((e) => String(e?.timestamp) === target);
  if (!entry) return null;
  const v = entry.value;
  const n = Number(typeof v === 'object' ? v?.value : v);
  return Number.isFinite(n) ? n : null;
}

function readTrafficSources(node) {
  const out = { foryou: null, follow: null, profile: null, search: null };
  const list = node?.value?.value ?? node?.value?.list ?? node?.value;
  if (!Array.isArray(list)) return out;
  for (const entry of list) {
    const key = (entry?.key ?? entry?.name ?? '').toString();
    const val = Number(entry?.value ?? entry?.percent ?? 0);
    if (!Number.isFinite(val)) continue;
    const norm = key.toLowerCase();
    if (norm === 'for you') out.foryou = val;
    else if (norm === 'follow') out.follow = val;
    else if (norm === 'personal profile') out.profile = val;
    else if (norm === 'search') out.search = val;
  }
  return out;
}

export function formatUnixDate(unixSeconds) {
  if (!unixSeconds) return '';
  const ms = unixSeconds > 1e12 ? unixSeconds : unixSeconds * 1000;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return '';
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

export function formatUnixTime(unixSeconds) {
  if (!unixSeconds) return '';
  const ms = unixSeconds > 1e12 ? unixSeconds : unixSeconds * 1000;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return '';
  const p = (n) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function roundTo(n, decimals) {
  const k = 10 ** decimals;
  return Math.round(n * k) / k;
}

const DEFAULT_INSIGHT_BASE_FOR_FOLLOWERS =
  'https://www.tiktok.com/aweme/v2/data/insight/?aid=1988&app_language=en&app_name=tiktok_creator_center&device_platform=web_pc&locale=en&channel=tiktok_web&os=mac';

const FOLLOWER_TYPE_REQUESTS = [
  { insigh_type: 'follower_num_history', days: 732, end_days: 1 },
  { insigh_type: 'follower_num',         days: 732, end_days: 1 },
  { insigh_type: 'net_follower_history', days: 732, end_days: 1 }
];

export function buildFollowerHistoryURL(template) {
  const base = template || DEFAULT_INSIGHT_BASE_FOR_FOLLOWERS;
  const sep = base.includes('?') ? '&' : '?';
  return `${base}${sep}type_requests=${encodeURIComponent(JSON.stringify(FOLLOWER_TYPE_REQUESTS))}`;
}

export function mapIndexToDate(i, _length, anchorIndex, anchorDate) {
  const delta = i - anchorIndex;
  const d = new Date(Date.UTC(
    anchorDate.getUTCFullYear(),
    anchorDate.getUTCMonth(),
    anchorDate.getUTCDate() + delta
  ));
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}`;
}
