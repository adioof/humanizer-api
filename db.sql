-- aucto.ai database schema (Supabase/Postgres)

-- API keys
create table api_keys (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users(id) on delete cascade not null,
  key_hash text not null unique,        -- SHA-256 of the actual key
  key_prefix text not null,             -- first 8 chars for display (ak_xxxxxxxx...)
  name text default 'Default',
  active boolean default true,
  created_at timestamptz default now(),
  last_used_at timestamptz
);

-- Credits / balance
create table credits (
  user_id uuid references auth.users(id) on delete cascade primary key,
  balance integer default 0 not null,   -- words remaining
  total_purchased integer default 0,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Usage log
create table usage_log (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users(id) on delete cascade not null,
  api_key_id uuid references api_keys(id) on delete set null,
  words_in integer not null,
  words_out integer not null,
  chunks integer,
  llm_model text,
  detection_score real,
  time_ms integer,
  created_at timestamptz default now()
);

-- Stripe customers
create table stripe_customers (
  user_id uuid references auth.users(id) on delete cascade primary key,
  stripe_customer_id text not null unique,
  created_at timestamptz default now()
);

-- Payment history
create table payments (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users(id) on delete cascade not null,
  stripe_payment_id text unique,
  amount_cents integer not null,
  words_credited integer not null,
  status text default 'pending',
  created_at timestamptz default now()
);

-- RLS policies
alter table api_keys enable row level security;
alter table credits enable row level security;
alter table usage_log enable row level security;
alter table stripe_customers enable row level security;
alter table payments enable row level security;

-- Users can only see their own data
create policy "Users see own keys" on api_keys for all using (auth.uid() = user_id);
create policy "Users see own credits" on credits for all using (auth.uid() = user_id);
create policy "Users see own usage" on usage_log for all using (auth.uid() = user_id);
create policy "Users see own stripe" on stripe_customers for all using (auth.uid() = user_id);
create policy "Users see own payments" on payments for all using (auth.uid() = user_id);

-- Index for fast API key lookup
create index idx_api_keys_hash on api_keys(key_hash) where active = true;
create index idx_usage_log_user on usage_log(user_id, created_at desc);

-- Auto-create credits row on signup
create or replace function handle_new_user()
returns trigger as $$
begin
  insert into credits (user_id, balance) values (new.id, 1000); -- 1000 free words
  return new;
end;
$$ language plpgsql security definer;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function handle_new_user();
