create table if not exists public.reddit_chatter (
    reddit_id text primary key,
    title text not null,
    score integer not null default 0,
    post_url text not null,
    snippet text not null default '',
    published_at timestamptz not null,
    fetched_at timestamptz not null default now(),
    raw_data jsonb not null default '{}'::jsonb
);

create index if not exists reddit_chatter_published_at_idx
    on public.reddit_chatter (published_at desc);
