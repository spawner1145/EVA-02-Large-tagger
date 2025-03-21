import asyncio
import httpx
import os
import re

# 评级映射
rating_mapping = {
    "g": "general",
    "s": "sensitive",
    "q": "questionable",
    "e": "explicit"
}

# 确保 dataset 目录存在
if not os.path.exists('dataset'):
    os.makedirs('dataset')

async def download_image(client, url, filename):
    try:
        response = await client.get(url)
        if response.status_code == 200:
            with open(os.path.join('dataset', filename), 'wb') as f:
                f.write(response.content)
            print(f"下载成功: {filename}")
        else:
            print(f"下载失败: {filename}, 状态码: {response.status_code}")
    except Exception as e:
        print(f"下载失败: {filename}, 错误: {e}")

async def process_post(client, post):
    rating = rating_mapping.get(post['rating'], post['rating'])
    tag_string = post['tag_string'].replace(" ", ",")
    metadata = f"{rating},{tag_string}"

    for variant in post['media_asset']['variants']:
        if variant['type'] == "720x720":
            image_url = variant['url']
            md5 = post['md5']
            image_filename = f"{md5}.{variant['file_ext']}"
            txt_filename = f"{md5}.txt"

            # 下载图片
            await download_image(client, image_url, image_filename)

            # 写入元数据到 txt 文件
            with open(os.path.join('dataset', txt_filename), 'w', encoding='utf-8') as f:
                f.write(metadata)
            break

async def fetch_posts(client, page, target_count, current_count):
    url = f"https://kagamihara.donmai.us/posts.json?page={page}"
    try:
        response = await client.get(url)
        if response.status_code == 200:
            posts = response.json()
            for post in posts:
                if current_count < target_count:
                    await process_post(client, post)
                    current_count += 1
                else:
                    break
            if current_count < target_count and posts:
                return await fetch_posts(client, page + 1, target_count, current_count)
            return current_count
        else:
            print(f"请求失败: {url}, 状态码: {response.status_code}")
    except Exception as e:
        print(f"请求失败: {url}, 错误: {e}")
    return current_count

async def main(target_count):
    async with httpx.AsyncClient() as client:
        total_downloaded = await fetch_posts(client, 1, target_count, 0)
        print(f"总共下载了 {total_downloaded} 张图片。")

if __name__ == "__main__":
    target_count = 10  # 你可以修改这个值来指定要爬取的图片数量
    asyncio.run(main(target_count))
    