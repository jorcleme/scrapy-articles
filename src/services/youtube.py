from turtle import title
from googleapiclient.discovery import build
import pandas as pd
import os
import json
import time
from typing import List, Dict, Any, Annotated
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    VideoUnplayable,
    NotTranslatable,
    CouldNotRetrieveTranscript,
    YouTubeRequestFailed,
    RequestBlocked,
    IpBlocked,
)
from datetime import datetime
from langchain_community.document_loaders.youtube import YoutubeLoader
from youtube_transcript_api.proxies import WebshareProxyConfig
from config import VIDEOS_DATA_DIR
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    field_serializer,
    field_validator,
    ConfigDict,
)
from datetime import datetime
from typing import Optional, Union
import re


def ensure_list(value: Any):
    """Ensure the value is a list."""
    if not isinstance(value, list):
        return [value]
    return value


class YoutubeVideo(BaseModel):
    """
    Pydantic model representing a YouTube video with comprehensive validation and serialization.
    """

    title: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1000,
            description="Title of the video",
            default="No title detected",
        ),
    ]
    # Basic video information
    video_id: Annotated[
        str, Field(..., min_length=11, max_length=11, description="YouTube video ID")
    ]

    # URLs and media
    url: Annotated[str, Field(..., description="Video embed URL")]

    # Dates and duration
    published_date: Annotated[
        str, Field(..., description="Video publication date in ISO format")
    ]
    duration: Annotated[
        Union[str, int],
        Field(..., description="Video duration in ISO 8601 format or seconds"),
    ]

    # Content
    description: Annotated[str, Field(default="", description="Video description")]
    tags: Annotated[List[str], Field(default_factory=list, description="Video tags")]
    transcript: Annotated[
        Union[str, None],
        Field(default=None, description="Video transcript or error message"),
    ]

    # Statistics (YouTube API returns these as strings)
    views: Annotated[int, Field(default=0, description="Number of views")]
    likes: Annotated[int, Field(default=0, description="Number of likes")]
    favorites: Annotated[int, Field(default=0, description="Number of favorites")]
    comments: Annotated[int, Field(default=0, description="Number of comments")]

    # Metadata
    kind: Annotated[str, Field(default="youtube", description="Content type")]
    category: Annotated[
        Union[str, None],
        Field(description="Resolved video category", default="Configuration"),
    ]
    series: Annotated[List[str], BeforeValidator(ensure_list)] = Field(
        default_factory=list, description="Resolved video series"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    @field_validator("views", "likes", "favorites", "comments")
    @classmethod
    def convert_to_int(cls, v: Union[str, int]) -> int:
        """Convert string numbers to integers."""
        try:
            return int(v) if v else 0
        except (ValueError, TypeError):
            return 0

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate and normalize YouTube URL."""
        if not any(domain in v for domain in ["youtube.com", "youtu.be"]):
            raise ValueError(f"Invalid YouTube URL: {v}")
        return v

    @field_validator("published_date")
    @classmethod
    def validate_published_date(cls, v: str) -> str:
        """Validate ISO date format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"Invalid date format. Expected ISO format, got: {v}")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Clean and validate tags."""
        if not isinstance(v, list):
            return []
        # Filter out empty strings and strip whitespace
        return [tag.strip() for tag in v if tag and tag.strip()]

    @field_validator("transcript")
    @classmethod
    def validate_transcript(cls, v: Union[str, None]) -> Union[str, None]:
        """Validate transcript field."""
        if v is None:
            return None
        if isinstance(v, str) and len(v.strip()) == 0:
            return None
        return v

    @field_serializer("series")
    def serialize_series(
        self, value: Union[str, List[str], None]
    ) -> Union[List[str], None]:
        """Ensure series is always a list or None."""
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        return None

    def get_duration_seconds(self) -> Optional[int]:
        """
        Convert ISO 8601 duration to seconds.

        Returns:
            int: Duration in seconds, or None if conversion fails
        """
        if isinstance(self.duration, int):
            return self.duration

        if isinstance(self.duration, str) and self.duration.startswith("PT"):
            # Parse ISO 8601 duration format (PT4M41S)
            pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
            match = re.match(pattern, self.duration)
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                return hours * 3600 + minutes * 60 + seconds

        return None

    def get_engagement_rate(self) -> float:
        """
        Calculate engagement rate as (likes + comments) / views.

        Returns:
            float: Engagement rate as a percentage
        """
        if self.views == 0:
            return 0.0

        total_engagement = self.likes + self.comments
        return (total_engagement / self.views) * 100

    def has_transcript(self) -> bool:
        """
        Check if video has a valid transcript.

        Returns:
            bool: True if transcript exists and is not an error message
        """
        if not self.transcript:
            return False

        error_indicators = [
            "Transcripts Disabled",
            "No English Transcript",
            "Video Unavailable",
            "Request Blocked",
            "Could Not Retrieve",
            "Not Translatable",
            "Error:",
        ]

        return not any(indicator in self.transcript for indicator in error_indicators)

    def get_transcript_word_count(self) -> int:
        """
        Get word count of transcript.

        Returns:
            int: Number of words in transcript
        """
        if not self.has_transcript():
            return 0

        return len(self.transcript.split())

    def __str__(self) -> str:
        return f"YouTubeVideo(id={self.video_id}, title='{self.title[:50]}...', views={self.views})"

    def __repr__(self) -> str:
        return self.__str__()

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)


class CiscoYouTubeDataLoader:
    def __init__(
        self,
        api_key: str,
        playlist_id: str,
        channel_ids: List[str],
        fetch: bool = True,
    ):
        """
        Initializes the CiscoYouTubeDataLoader class with the necessary API key, playlist ID, and channel IDs.

        :param api_key: A valid Google API key.
        :param playlist_id: The ID of the YouTube playlist to process.
        :param channel_ids: A list of YouTube channel IDs to gather stats for.
        """
        self.playlist_id = playlist_id
        self.channel_ids = channel_ids
        self.youtube_service = build("youtube", "v3", developerKey=api_key)
        self.scraped_videos_json = self._load_scraped_videos()

        if fetch:
            self._videos = []
            self._fetch_video_data()

        elif self.scraped_videos_json:
            self._videos = self.scraped_videos_json
            videos_with_transcripts = self.get_video_transcript(self._videos)
            videos_with_categories = self.resolve_category(videos_with_transcripts)
            videos_with_series = self.resolve_to_series(videos_with_categories)
            self._videos = videos_with_series

        else:
            raise ValueError(
                f"No video data found on filesystem and fetch is set to '{fetch}'. Set fetch keyword to 'True' to fetch new data."
            )

    def _fetch_video_data(self):
        playlist_video_ids = self.get_video_ids_by_playlist_id()
        video_data = self.get_video_data(playlist_video_ids)
        videos_with_transcripts = self.get_video_transcript(video_data)
        videos_with_categories = self.resolve_category(videos_with_transcripts)
        videos_with_series = self.resolve_to_series(videos_with_categories)
        self._videos = videos_with_series

    @property
    def videos(self):
        return self._videos

    @videos.setter
    def videos(self, value):
        self._videos = value

    @staticmethod
    def _load_scraped_videos() -> List[YoutubeVideo]:
        """
        Loads previously scraped video data from a JSON file.

        Returns:
            List: Returns the list of video data from the JSON file or an empty list if the file is not found.
        """
        try:
            with open(f"{VIDEOS_DATA_DIR}/youtube_smbvideos.json", "r") as file:
                data = json.load(file)
                return [YoutubeVideo.model_validate(video) for video in data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _get_previously_scraped_video(self, video_id: str):
        for video in self.scraped_videos_json:
            if video.video_id == video_id:
                return video.model_dump()

    def _has_previously_scraped_video(self, video_id: str) -> bool:
        for video in self.scraped_videos_json:
            if video.video_id == video_id:
                return True
        return False

    def get_channel_stats_by_id(self):
        """
        Retrieves the statistics for the channels associated with the provided channel IDs.

        Returns:
            List: Returns a list of dictionaries containing the channel statistics.
        """
        stats = []
        request = self.youtube_service.channels().list(
            part="snippet,contentDetails,statistics", id=",".join(self.channel_ids)
        )
        response = request.execute()

        for i in range(len(response["items"])):
            data = dict(
                channel_name=response["items"][i]["snippet"]["title"],
                subscribers=response["items"][i]["statistics"]["subscriberCount"],
                videos=response["items"][i]["statistics"]["videoCount"],
                views=response["items"][i]["statistics"]["viewCount"],
                playlist_id=response["items"][i]["contentDetails"]["relatedPlaylists"][
                    "uploads"
                ],
            )
            stats.append(data)
        return stats

    def get_video_ids_by_playlist_id(self):
        """
        Retrieves the video IDs from the specified playlist.

        Returns:
            List: Returns a list of video IDs from the playlist.
        """
        videos = []
        request = self.youtube_service.playlistItems().list(
            part="contentDetails", playlistId=self.playlist_id, maxResults=50
        )
        response = request.execute()

        for i in range(len(response["items"])):
            video_id = response["items"][i]["contentDetails"]["videoId"]
            videos.append(video_id)
        next_page_token = response.get("nextPageToken")

        while next_page_token:
            request = self.youtube_service.playlistItems().list(
                part="contentDetails",
                playlistId=self.playlist_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            response = request.execute()
            for i in range(len(response["items"])):
                video_id = response["items"][i]["contentDetails"]["videoId"]
                videos.append(video_id)
            next_page_token = response.get("nextPageToken")

        return videos

    def get_video_data(self, video_ids: List[str]):
        """
        Retrieves the data for the specified video IDs.

        Args:
            video_ids (List[str]): A list of video IDs to retrieve data for.

        Returns:
            List: Returns a list of dictionaries containing the video data.
        """
        videos = []

        try:
            for i in range(0, len(video_ids), 50):
                request = self.youtube_service.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=",".join(video_ids[i : i + 50]),
                )
                response = request.execute()
                for video in response["items"]:
                    id = video["id"]

                    data = dict(
                        title=video["snippet"]["title"],
                        published_date=video["snippet"]["publishedAt"],
                        description=video["snippet"]["description"],
                        url=f"https://www.youtube.com/embed/{id}",
                        video_id=id,
                        views=video["statistics"]["viewCount"],
                        likes=video["statistics"]["likeCount"],
                        favorites=video["statistics"]["favoriteCount"],
                        duration=video["contentDetails"]["duration"],
                        comments=video["statistics"]["commentCount"],
                        kind=video["kind"].split("#")[0] or "youtube",
                        tags=video["snippet"].get("tags", []),
                    )

                    if self._has_previously_scraped_video(video_id=id):
                        stale_video_data = self._get_previously_scraped_video(
                            video_id=id
                        )
                        # update stale_data and spread fetched data as likes, comments and views could have changed.
                        stale_video_data.update(data)
                        videos.append(stale_video_data)
                    else:
                        videos.append(data)
        except Exception as e:
            print(e)

        return [YoutubeVideo.model_validate(video) for video in videos]

    def get_pydantic_videos(self) -> List[YoutubeVideo]:
        """
        Get all processed videos as validated Pydantic models.

        Returns:
            List of YoutubeVideo Pydantic models with validation and utility methods
        """
        return [YoutubeVideo.model_validate(video) for video in self._videos]

    def export_pydantic_videos(self, filepath: str) -> None:
        """
        Export videos as validated Pydantic models to JSON.

        Args:
            filepath: Path to save the JSON file
        """
        pydantic_videos = self.get_pydantic_videos()
        validated_data = [video.model_dump() for video in pydantic_videos]

        with open(filepath, "w") as f:
            json.dump(validated_data, f, indent=2, default=str)

        print(f"💾 Exported {len(validated_data)} validated videos to {filepath}")

    def get_video_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics using Pydantic models.

        Returns:
            Dictionary containing analytics data
        """
        videos = self.get_pydantic_videos()

        if not videos:
            return {"error": "No videos available for analysis"}

        total_videos = len(videos)
        total_views = sum(v.views for v in videos)
        total_likes = sum(v.likes for v in videos)
        total_comments = sum(v.comments for v in videos)
        videos_with_transcripts = sum(1 for v in videos if v.has_transcript())

        durations = [
            v.get_duration_seconds() for v in videos if v.get_duration_seconds()
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0

        engagement_rates = [v.get_engagement_rate() for v in videos]
        avg_engagement = (
            sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0
        )

        # Category breakdown
        categories = {}
        for video in videos:
            if video.category:
                categories[video.category] = categories.get(video.category, 0) + 1

        # Series breakdown
        series_count = {}
        for video in videos:
            if video.series:
                series_list = (
                    video.series if isinstance(video.series, list) else [video.series]
                )
                for series in series_list:
                    series_count[series] = series_count.get(series, 0) + 1

        return {
            "total_videos": total_videos,
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "videos_with_transcripts": videos_with_transcripts,
            "transcript_percentage": (videos_with_transcripts / total_videos) * 100,
            "average_duration_seconds": avg_duration,
            "average_engagement_rate": avg_engagement,
            "categories": categories,
            "series": series_count,
            "top_performing_videos": sorted(
                [(v.title, v.views, v.get_engagement_rate()) for v in videos],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }

    def get_video_transcript(self, video_data: List[YoutubeVideo]):
        videos = []
        videos_data = [video.model_dump() for video in video_data]
        transcript_stats = {
            "total": 0,
            "successful": 0,
            "disabled": 0,
            "not_found": 0,
            "unavailable": 0,
            "rate_limited": 0,
            "other_errors": 0,
            "skipped_existing": 0,
        }

        for video in videos_data:
            transcript_stats["total"] += 1

            if "transcript" in video and video["transcript"] is not None:
                videos.append(video)
                transcript_stats["skipped_existing"] += 1
                print(
                    f"⏭️  Skipping video ID {video['video_id']} as transcript already exists."
                )
                continue

            video_id = video["video_id"]

            try:
                # Wrap the entire transcript fetching process in try-catch
                yt_api = YouTubeTranscriptApi(
                    proxy_config=WebshareProxyConfig(
                        proxy_username=os.getenv("PROXY_USER_NAME"),
                        proxy_password=os.getenv("PROXY_PASSWORD"),
                    )
                )
                transcript_list = yt_api.list(video_id=video_id)
                transcript = transcript_list.find_transcript(["en"]).fetch()

                # Successfully got transcript
                video["transcript"] = " ".join(
                    [line.text if line else "" for line in transcript]
                )
                transcript_stats["successful"] += 1
                print(f"✅ Successfully fetched transcript for video ID {video_id}")

            except TranscriptsDisabled:
                print(f"⚠️  Transcripts disabled for video ID {video_id}")
                video["transcript"] = "Transcripts Disabled"
                transcript_stats["disabled"] += 1
            except NoTranscriptFound:
                print(f"⚠️  No English transcript found for video ID {video_id}")
                video["transcript"] = "No English Transcript"
                transcript_stats["not_found"] += 1
            except (VideoUnavailable, VideoUnplayable):
                print(f"⚠️  Video unavailable/unplayable for video ID {video_id}")
                video["transcript"] = "Video Unavailable"
                transcript_stats["unavailable"] += 1
            except (RequestBlocked, IpBlocked, YouTubeRequestFailed):
                print(
                    f"⚠️  Request blocked/failed for video ID {video_id} - likely rate limited. Waiting 10 seconds..."
                )
                video["transcript"] = "Request Blocked"
                transcript_stats["rate_limited"] += 1
                time.sleep(10)  # Wait longer for blocked requests
            except CouldNotRetrieveTranscript:
                print(f"⚠️  Could not retrieve transcript for video ID {video_id}")
                video["transcript"] = "Could Not Retrieve"
                transcript_stats["other_errors"] += 1
            except NotTranslatable:
                print(f"⚠️  Transcript not translatable for video ID {video_id}")
                video["transcript"] = "Not Translatable"
                transcript_stats["other_errors"] += 1
            except Exception as e:
                # Handle any other unexpected exceptions
                error_type = type(e).__name__
                print(
                    f"❌ Unexpected error fetching transcript for video ID {video_id}. Error {error_type}: {e}"
                )
                video["transcript"] = f"Error: {error_type}"
                transcript_stats["other_errors"] += 1

            # Always append the video, whether transcript was successful or not
            videos.append(video)

            # Small delay between requests to be respectful
            if transcript_stats["total"] % 10 == 0:
                print(
                    f"🔄 Processed {transcript_stats['total']} videos, taking a short break..."
                )
                time.sleep(2)

        # Print summary statistics
        print("\n📊 Transcript Fetch Summary:")
        print(f"   Total videos processed: {transcript_stats['total']}")
        print(f"   ✅ Successful: {transcript_stats['successful']}")
        print(f"   ⏭️  Skipped (existing): {transcript_stats['skipped_existing']}")
        print(f"   ⚠️  Transcripts disabled: {transcript_stats['disabled']}")
        print(f"   ⚠️  No English transcript: {transcript_stats['not_found']}")
        print(f"   ⚠️  Video unavailable: {transcript_stats['unavailable']}")
        print(f"   ⚠️  Rate limited: {transcript_stats['rate_limited']}")
        print(f"   ❌ Other errors: {transcript_stats['other_errors']}")

        success_rate = (
            transcript_stats["successful"]
            / max(1, transcript_stats["total"] - transcript_stats["skipped_existing"])
        ) * 100
        print(f"   📈 Success rate: {success_rate:.1f}%")
        print(f"   🎯 Total videos with data: {len(videos)}")

        return [YoutubeVideo.model_validate(video) for video in videos]

    def resolve_category(self, video_data: List[YoutubeVideo]):
        """
        Resolves the category of the videos based on the title, description, and tags.

        Args:
            video_data (List[Dict[str, Dict[str, Any]]]): A list of dictionaries containing the video data.

        Returns:
            List: Returns a list of dictionaries containing the video data with the resolved category.
        """
        videos_with_categories = []

        videos = [video.model_dump() for video in video_data]

        for video in videos:
            video_id = video["video_id"]

            title = video["title"]
            description = video["description"].lower()
            tags = [tag.lower() for tag in video["tags"]]
            video["category"] = "Configuration"

            # Configuration
            configuring = [
                "firewall",
                "configure",
                "tech talk",
                "cisco tech talk",
                "bluetooth",
                "configurations",
                "configuring",
                "configuration",
                "deploy",
            ]
            if (
                any(word in title for word in configuring)
                or any(word in description for word in configuring)
                or any(word in tags for word in configuring)
            ):
                video["category"] = "Configuration"
                # Install & Upgrade
            install_upgrade = [
                "install",
                "upgrade",
                "installation",
                "upgrading",
                "day 0",
                "day",
                "get to know",
                "getting to know",
                "get-to-know",
            ]
            if any(word in title for word in install_upgrade) or any(
                word in description for word in install_upgrade
            ):
                video["category"] = "Install & Upgrade"
                # Maintain & Operate
            maintain_operate = [
                "reboot",
                "restarting",
                "restart",
                "rebooting",
                "rebooted",
                "restarting",
                "maintain",
                "operate",
                "cli",
                "command line",
                "command-line",
                "command line interface",
                "terminal",
            ]
            if any(word in title for word in maintain_operate) or any(
                word in description for word in maintain_operate
            ):
                video["category"] = "Maintain & Operate"
            # Troubleshooting
            troubleshooting = [
                "troubleshoot",
                "troubleshooting",
                "troubleshooter",
                "troubleshooters",
                "tips",
            ]
            if any(word in title for word in troubleshooting) or any(
                word in description for word in troubleshooting
            ):
                video["category"] = "Troubleshooting"
            # Design
            design = [
                "design",
                "designing",
                "designs",
                "new to cisco",
                "cisco business",
            ]
            if any(word in title for word in design) or any(
                word in description for word in design
            ):
                video["category"] = "Design"

            videos_with_categories.append(video)

        return [YoutubeVideo.model_validate(video) for video in videos_with_categories]

        for video in video_data:
            video_id = video.video_id
            video_details = {video_id: video}
            videos.append(video_details)

        for video in videos:
            for k, v in video.items():
                title = v.title.lower()
                description = v.description.lower()
                tags = [tag.lower() for tag in v.tags]
                video[k].category = "Configuration"

                # Configuration
                configuring = [
                    "firewall",
                    "configure",
                    "tech talk",
                    "cisco tech talk",
                    "bluetooth",
                    "configurations",
                    "configuring",
                    "configuration",
                    "deploy",
                ]
                if (
                    any(word in title for word in configuring)
                    or any(word in description for word in configuring)
                    or any(word in tags for word in configuring)
                ):
                    video[k]["category"] = "Configuration"
                # Install & Upgrade
                install_upgrade = [
                    "install",
                    "upgrade",
                    "installation",
                    "upgrading",
                    "day 0",
                    "day",
                    "get to know",
                    "getting to know",
                    "get-to-know",
                ]
                if any(word in title for word in install_upgrade) or any(
                    word in description for word in install_upgrade
                ):
                    video[k].category = "Install & Upgrade"
                # Maintain & Operate
                maintain_operate = [
                    "reboot",
                    "restarting",
                    "restart",
                    "rebooting",
                    "rebooted",
                    "restarting",
                    "maintain",
                    "operate",
                    "cli",
                    "command line",
                    "command-line",
                    "command line interface",
                    "terminal",
                ]
                if any(word in title for word in maintain_operate) or any(
                    word in description for word in maintain_operate
                ):
                    video[k].category = "Maintain & Operate"
                # Troubleshooting
                troubleshooting = [
                    "troubleshoot",
                    "troubleshooting",
                    "troubleshooter",
                    "troubleshooters",
                    "tips",
                ]
                if any(word in title for word in troubleshooting) or any(
                    word in description for word in troubleshooting
                ):
                    video[k].category = "Troubleshooting"
                # Design
                design = [
                    "design",
                    "designing",
                    "designs",
                    "new to cisco",
                    "cisco business",
                ]
                if any(word in title for word in design) or any(
                    word in description for word in design
                ):
                    video[k]["category"] = "Design"

            videos_with_categories.append(video)

        return videos_with_categories

    def resolve_to_series(self, videos: List[YoutubeVideo]):
        """
        Resolves the series of the videos based on the tags.

        Args:
            videos (List[YoutubeVideo]): A list of YoutubeVideo objects containing the video data.

        Returns:
            List: Returns a list of dictionaries containing the video data with the resolved series.
        """

        videos_data = [video.model_dump() for video in videos]
        videos_with_series = []
        for video in videos_data:
            series = video.get("series")

            if series and not isinstance(series, list):
                video["series"] = [series]

            tags = [tag.lower() for tag in video["tags"]]
            title = video["title"].lower()
            catalyst_1200_series = [
                "c1200",
                "1200",
                "cat1200",
                "catalyst 1200",
                "qos",
                "eol",
                "1200/1300",
            ]
            if any(word in tags for word in catalyst_1200_series) or any(
                word in title
                for word in ["catalyst 1200", "c1200", "cat1200", "catalyst 1200/1300"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Catalyst 1200 Series Switches")

            catalyst_1300_series = [
                "c1300",
                "1300",
                "cat1300",
                "catalyst 1300",
                "qos",
                "eol",
                "1200/1300",
            ]
            if any(word in tags for word in catalyst_1300_series) or any(
                word in title
                for word in ["catalyst 1300", "c1300", "cat1300", "catalyst 1200/1300"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Catalyst 1300 Series Switches")

            cbs110_series = ["cbs110", "eol", "110", "unmanaged"]
            if any(word in tags for word in cbs110_series) or any(
                word in title
                for word in ["cbs110", "cbs 110", "cisco business 110 unmanaged switch"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business 110 Series Unmanaged Switches")

            cbs220_series = ["cbs220", "qos", "cisco business 220", "eol"]
            if any(word in tags for word in cbs220_series) or any(
                word in title
                for word in [
                    "cbs220",
                    "cbs 220",
                    "cisco business 220 series switches",
                    "cbs switch",
                    "cisco business switches",
                    "cisco business switch",
                ]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business 220 Series Smart Switches")

            cbs250_series = [
                "cbs250",
                "cisco business 250",
                "qos",
                "eol",
                "250",
                "cbs250/350",
            ]
            if any(word in tags for word in cbs250_series) or any(
                word in title
                for word in [
                    "cbs250",
                    "cbs250/350",
                    "cbs 250",
                    "cbs switch",
                    "cisco business switches",
                    "cisco business switch",
                ]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business 250 Series Smart Switches")

            cbs350_series = [
                "cbs350",
                "cisco business 350",
                "qos",
                "eol",
                "350",
                "cbs250/350",
                "3504x",
            ]
            if any(word in tags for word in cbs350_series) or any(
                word in title
                for word in [
                    "cbs350",
                    "cbs250/350",
                    "cbs 350",
                    "cbs switch",
                    "cisco business switches",
                    "cisco business switch",
                ]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business 350 Series Managed Switches")

            smb350_series = [
                "sf350",
                "sg350",
                "sg 350 series",
                "eol",
                "ciscosx350xguide",
            ]
            if any(word in tags for word in smb350_series) or any(
                word in title for word in ["smb350", "sg350"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco 350 Series Managed Switches")

            # SMB350X
            smb350x_series = [
                "sf350",
                "sg350x",
                "sg 350x series",
                "sg350x",
                "sg350xg",
                "sf",
                "sg",
                "qos",
                "eol",
                "sg300",
                "sf300",
                "ciscosx350xguide",
            ]

            if any(word in tags for word in smb350x_series):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco 350X Series Managed Switches")

            smb550x_series = [
                "sf550",
                "sg550x",
                "sg 550x series",
                "sg550x",
                "sg550xg",
                "sf",
                "sg",
                "qos",
                "eol",
            ]
            if any(word in tags for word in smb550x_series):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco 550X Series Managed Switches")

            # RV100
            rv100_series = [
                "how to create a secure tunnel between two rv130w routers",
                "rv120w",
                "rv130",
                "rv130w",
                "rv130w router",
                "rv180",
                "rv180w",
                "eol",
                "small business router",
                "routers",
            ]
            if any(word in tags for word in rv100_series):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("RV100 Product Family")

            # RV160 VPN Router
            rv160_series = [
                "rv160",
                "rv160 rv260 rv series routers",
                "rv160/260",
                "cisco rv160",
                "cisco rv160 router",
                "smb-routers-rv160-series",
                "what is rv160 router",
                "eol",
                "small business router",
                "routers",
            ]
            if any(word in tags for word in rv160_series):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("RV160 VPN Router")

            # RV260 VPN Router
            rv260_series = [
                "cisco rv260",
                "cisco rv260 router",
                "rv160/260",
                "cisco rv260w",
                "cisco rv260w router",
                "how to configure rv260w router",
                "rv160 rv260 rv series routers",
                "rv260",
                "rv260 series",
                "rv260p",
                "rv260w",
                "rv260w router",
                "rv260w router set up",
                "rv260w router setup",
                "rv260w set up",
                "rv260w setup",
                "set up rv260w",
                "set up rv260w router",
                "setup rv260w router",
                "eol",
                "small business router",
                "routers",
            ]
            if any(word in tags for word in rv260_series) or any(
                word in title for word in ["rv260", "rv260w"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("RV260 VPN Router")

            # RV320 Series
            rv320_series = ["rv320", "smb-routers-rv320-series", "eol"]
            if any(word in tags for word in rv320_series) or any(
                word in title for word in ["rv320", "rv325"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("RV320 Product Family")

            # RV340 Series
            rv340_series = [
                "rv340",
                "smb-routers-rv340-series",
                "cisc rv340 router",
                "rv345 series routers",
                "cisco business rv340",
                "cisco rv340",
                "cisco rv340 router",
                "cisco rv340 router policy",
                "cisco rv340 series",
                "cisco rv340 series router",
                "cisco rv340 series routers",
                "cisco rv340w",
                "ciscorv340",
                "rv340 router",
                "rv340 series",
                "rv340 series router",
                "rv340w",
                "eol",
                "small business router",
                "routers",
            ]
            if any(word in tags for word in rv340_series) or any(
                word in title for word in ["rv340", "rv34x"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("RV340 Product Family")

            # CBW AC
            cbw_ac_series = [
                "cbw140",
                "cbw140ac",
                "cbw141",
                "cbw141acm",
                "cbw142",
                "cbw142acm",
                "143acm",
                "142acm",
                "cbw143acm",
                "cbw144ac",
                "cbw145",
                "cbw145ac",
                "cisco mesh",
                "cbw240",
                "cbw240ac",
                "eol",
                "140/145/240",
                "cisco140acaccesspoint",
                "ciscobusiness240acaccesspoint",
                "mesh",
                "wireless mesh",
            ]
            if any(word in tags for word in cbw_ac_series) or any(
                word in title
                for word in ["cbw140", "cbw141", "cbw142", "141acm", "cbw"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business Wireless AC")

            # CBW AX
            cbw_ax_series = [
                "cbw150",
                "cbw150ax",
                "cbw151",
                "cbw151ax",
                "cbw151axm",
                "eol",
                "mesh",
                "wireless mesh",
            ]
            if any(word in tags for word in cbw_ax_series) or any(
                word in title for word in ["cbw150", "cbw151", "mesh wi-fi", "cbw"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business Wireless AX")

            # WAP 100 Series
            wap_100_series = [
                "cisco wap125",
                "ciscowap150",
                "configuring wap",
                "wap125",
                "wap150",
                "wap150_indoor_wall_mounting",
                "wireless access network business cisco systems",
                "eol",
                "ap",
            ]
            if any(word in tags for word in wap_100_series) or any(
                word in title for word in ["wap125", "wap150", "wap125-581"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append(
                    "Cisco Small Business 100 Series Wireless Access Points"
                )

            # WAP 300 Series
            wap_300_series = [
                "how to manage channels on wap371",
                "wap 371",
                "wap361",
                "wap371",
                "wireless Access network business Cisco Systems",
                "eol",
                "ap",
            ]

            if any(word in tags for word in wap_300_series):
                if "series" not in video:
                    video["series"] = []
                video["series"].append(
                    "Cisco Small Business 300 Series Wireless Access Points"
                )

            # WAP 500 Series
            wap_500_series = [
                "cisco wap581",
                "wap571",
                "wap571_ceiling_mounting",
                "wap571_indoor_mounting_options",
                "wap571_wall_mounting",
                "wap571e",
                "wap581",
                "wireless Access network business Cisco Systems",
                "eol",
                "ap",
            ]
            if any(word in tags for word in wap_500_series) or any(
                word in title for word in ["wap581", "wap571", "wireless access point"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append(
                    "Cisco Small Business 500 Series Wireless Access Points"
                )

            # CP6800 Phones
            cp6800_series = [
                "cp-6800",
                "cisco multiplatform firmware",
                "multiplatform",
                "eol",
                "mpp",
            ]
            if any(word in tags for word in cp6800_series):
                if "series" not in video:
                    video["series"] = []
                video["series"].append(
                    "Cisco IP Phone 6800 Series with Multiplatform Firmware"
                )

            # CP7800 Phones
            cp7800_series = [
                "7800",
                "7800 cisco multiplatform phone",
                "cisco 7800 series ip phone",
                "cp-7800",
                "eol",
                "ip phone",
                "ip phones",
                "ciscospaseriesipphones",
                "multiplatform",
                "mpp",
            ]
            if any(word in tags for word in cp7800_series):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco IP Phone 7800 Series")

            # CP8800 Phones
            cp8800_series = [
                "8800",
                "8800 cisco multiplatform phone",
                "cisco 8800 series ip phone",
                "cp-8800",
                "cisco cp-8800",
                "8800",
                "8800 cisco multiplatform phone",
                "eol",
                "ip phone",
                "ip phones",
                "ciscospaseriesipphones",
                "8800 mpp phones",
                "multiplatform",
                "8865",
                "mpp",
                "8800 mpp phones",
            ]
            if any(word in tags for word in cp8800_series) or any(
                word in title for word in ["cp8800", "cisco 8800", "cisco 8800 series"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco IP Phone 8800 Series")

            # Cisco Business Dashboard
            business_dashboard = [
                "cbd",
                "cbd 2.3",
                "cbd licensing",
                "cbd probe",
                "business dashboard",
                "cisco business dashboard",
                "cisco business dashboard features",
                "cisco business dashboard network monitoring",
                "eol",
                "dashboard",
            ]
            if any(word in tags for word in business_dashboard):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business Dashboard")

            # Cisco Business Dashboard Lite
            cisco_business_dashboard_lite = ["cbd lite", "lite", "eol"]
            if any(word in tags for word in cisco_business_dashboard_lite):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business Dashboard Lite")

            # Cisco Business Mobile App
            business_mobile_app = [
                "cisco business mobile app",
                "cisco business mobile app features",
                "cisco business mobile app installation",
                "cisco business mobile app setup",
                "mobile",
                "mobile app",
                "mobile network",
                "mobile network settings",
                "eol",
            ]
            if any(word in tags for word in business_mobile_app):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business Mobile App")

            # Cisco Business FindIT
            business_findit = [
                "cisco findit",
                "cisco findit network management",
                "cisco findit topology",
                "findit",
                "eol",
                "findit network management",
            ]
            if any(word in tags for word in business_findit) or any(
                word in title for word in ["findit"]
            ):
                if "series" not in video:
                    video["series"] = []
                video["series"].append("Cisco Business FindIT")

            video["series"] = list(set(video.get("series", [])))
            print(video["series"])
            videos_with_series.append(video)

        # Ensure unique videos by video_id (keeps last occurrence)
        return [
            YoutubeVideo.model_validate(video)
            for video in {
                video["video_id"]: video for video in videos_with_series
            }.values()
        ]

        concepts = []
        for video in videos:
            for _, values in video.items():
                tags = [tag.lower() for tag in values["tags"]]

                # Catalyst 1200 Series
                catalyst_1200_series = ["c1200", "cat1200", "catalyst 1200"]
                if any(word in tags for word in catalyst_1200_series):
                    if "series" not in values:
                        values["series"] = []

                    if not isinstance(values["series"], list):
                        del values["series"]
                    values["series"].append("Cisco Catalyst 1200 Series Switches")
                    concepts.append(values.copy())

                # Catalyst 1300 Series
                catalyst_1300_series = ["c1300", "cat1300", "catalyst 1300"]
                if any(word in tags for word in catalyst_1300_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco Catalyst 1300 Series Switches")
                    concepts.append(values.copy())

                # CBS110
                cbs110_series = ["cbs110"]
                if any(word in tags for word in cbs110_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append(
                        "Cisco Business 110 Series Unmanaged Switches"
                    )
                    concepts.append(values.copy())

                # CBS220
                cbs220_series = ["cbs220"]
                if any(word in tags for word in cbs220_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco Business 220 Series Smart Switches")
                    concepts.append(values.copy())

                # CBS250
                cbs250_series = ["cbs250"]
                if any(word in tags for word in cbs250_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco Business 250 Series Smart Switches")
                    concepts.append(values.copy())

                # CBS350
                cbs350_series = ["cbs350"]
                if any(word in tags for word in cbs350_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append(
                        "Cisco Business 350 Series Managed Switches"
                    )
                    concepts.append(values.copy())

                # SMB350
                smb350_series = ["sf350", "sg350", "sg 350 series"]
                if any(word in tags for word in smb350_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco 350 Series Managed Switches")
                    concepts.append(values.copy())

                # SMB350X
                smb350x_series = [
                    "sf350",
                    "sg350x",
                    "sg 350x series",
                    "sg350x",
                    "sg350xg",
                    "sf",
                    "sg",
                ]
                if any(word in tags for word in smb350x_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append(
                        "Cisco 350X Series Stackable Managed Switches"
                    )
                    concepts.append(values.copy())

                # SMB550X
                smb550x_series = [
                    "550x",
                    "sf550x",
                    "sg550x",
                    "sg 550x series",
                    "sg500",
                    "sg500x",
                    "sg550",
                    "sg550 series small business switches",
                    "sg550 seriesz",
                    "sg550x",
                    "sg550xg",
                ]
                if any(word in tags for word in smb550x_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append(
                        "Cisco 550X Series Stackable Managed Switches"
                    )
                    concepts.append(values.copy())

                # RV100
                rv100_series = [
                    "how to create a secure tunnel between two rv130w routers",
                    "rv120w",
                    "rv130",
                    "rv130w",
                    "rv130w router",
                    "rv180",
                    "rv180w",
                ]
                if any(word in tags for word in rv100_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("RV100 Product Family")
                    concepts.append(values.copy())

                # RV160 VPN Router
                rv160_series = [
                    "rv160",
                    "rv160 rv260 rv series routers",
                    "cisco rv160",
                    "cisco rv160 router",
                    "smb-routers-rv160-series",
                    "what is rv160 router",
                ]
                if any(word in tags for word in rv160_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("RV160 VPN Router")
                    concepts.append(values.copy())

                # RV260 VPN Router
                rv260_series = [
                    "cisco rv260",
                    "cisco rv260 router",
                    "cisco rv260w",
                    "cisco rv260w router",
                    "how to configure rv260w router",
                    "rv160 rv260 rv series routers",
                    "rv260",
                    "rv260 series",
                    "rv260p",
                    "rv260w",
                    "rv260w router",
                    "rv260w router set up",
                    "rv260w router setup",
                    "rv260w set up",
                    "rv260w setup",
                    "set up rv260w",
                    "set up rv260w router",
                    "setup rv260w router",
                ]
                if any(word in tags for word in rv260_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("RV260 VPN Router")
                    concepts.append(values.copy())

                # RV320 Series
                rv320_series = ["rv320", "smb-routers-rv320-series"]
                if any(word in tags for word in rv320_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("RV320 Product Family")
                    concepts.append(values.copy())

                # RV340 Series
                rv340_series = [
                    "rv340",
                    "smb-routers-rv340-series",
                    "cisc rv340 router",
                    "cisco business rv340",
                    "cisco rv340",
                    "cisco rv340 router",
                    "cisco rv340 router policy",
                    "cisco rv340 series",
                    "cisco rv340 series router",
                    "cisco rv340 series routers",
                    "cisco rv340w",
                    "ciscorv340",
                    "rv340 router",
                    "rv340 series",
                    "rv340 series router",
                    "rv340w",
                ]
                if any(word in tags for word in rv340_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("RV340 Product Family")
                    concepts.append(values.copy())

                # CBW AC
                cbw_ac_series = [
                    "cbw140",
                    "cbw140ac",
                    "cbw141",
                    "cbw141acm",
                    "cbw142",
                    "cbw142acm",
                    "cbw143acm",
                    "cbw144ac",
                    "cbw145",
                    "cbw145ac",
                    "cisco mesh",
                    "cbw240",
                    "cbw240ac",
                ]
                if any(word in tags for word in cbw_ac_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco Business Wireless AC")
                    concepts.append(values.copy())

                # CBW AX
                cbw_ax_series = [
                    "cbw150",
                    "cbw150ax",
                    "cbw151",
                    "cbw151ax",
                    "cbw151axm",
                ]
                if any(word in tags for word in cbw_ax_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco Business Wireless AX")
                    concepts.append(values.copy())
                # WAP 100 Series
                wap_100_series = [
                    "cisco wap125",
                    "ciscowap150",
                    "configuring wap",
                    "wap125",
                    "wap150",
                    "wap150_indoor_wall_mounting",
                ]
                if any(word in tags for word in wap_100_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append(
                        "Cisco Small Business 100 Series Wireless Access Points"
                    )
                    concepts.append(values.copy())

                # WAP 300 Series
                wap_300_series = [
                    "how to manage channels on wap371",
                    "wap 371",
                    "wap361",
                    "wap371",
                ]

                if any(word in tags for word in wap_300_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append(
                        "Cisco Small Business 300 Series Wireless Access Points"
                    )
                    concepts.append(values.copy())

                # WAP 500 Series
                wap_500_series = [
                    "cisco wap581",
                    "wap571",
                    "wap571_ceiling_mounting",
                    "wap571_indoor_mounting_options",
                    "wap571_wall_mounting",
                    "wap571e",
                    "wap581",
                ]
                if any(word in tags for word in wap_500_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append(
                        "Cisco Small Business 500 Series Wireless Access Points"
                    )
                    concepts.append(values.copy())

                # CP6800 Phones
                cp6800_series = ["cp-6800", "cisco multiplatform firmware"]
                if any(word in tags for word in cp6800_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append(
                        "Cisco IP Phone 6800 Series with Multiplatform Firmware"
                    )
                    concepts.append(values.copy())

                # CP7800 Phones
                cp7800_series = [
                    "7800",
                    "7800 cisco multiplatform phone",
                    "cisco 7800 series ip phone",
                    "cp-7800",
                ]
                if any(word in tags for word in cp7800_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco IP Phone 7800 Series")
                    concepts.append(values.copy())

                # CP8800 Phones
                cp8800_series = [
                    "8800",
                    "8800 cisco multiplatform phone",
                    "cisco 8800 series ip phone",
                    "cp-8800",
                    "cisco cp-8800",
                    "8800",
                    "8800 cisco multiplatform phone",
                ]
                if any(word in tags for word in cp8800_series):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco IP Phone 8800 Series")
                    concepts.append(values.copy())

                # Cisco Business Dashboard
                business_dashboard = [
                    "cbd",
                    "cbd 2.3",
                    "cbd licensing",
                    "cbd probe",
                    "business dashboard",
                    "cisco business dashboard",
                    "cisco business dashboard features",
                    "cisco business dashboard network monitoring",
                ]
                if any(word in tags for word in business_dashboard):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco Business Dashboard")
                    concepts.append(values.copy())

                # Cisco Business Mobile App
                business_mobile_app = [
                    "cisco business mobile app",
                    "cisco business mobile app features",
                    "cisco business mobile app installation",
                    "cisco business mobile app setup",
                    "mobile",
                    "mobile app",
                    "mobile network",
                    "mobile network settings",
                ]

                if any(word in tags for word in business_mobile_app):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco Business Mobile App")
                    concepts.append(values.copy())

                # Cisco Business FindIT
                business_findit = [
                    "cisco findit",
                    "cisco findit network management",
                    "cisco findit topology",
                ]
                if any(word in tags for word in business_findit):
                    if "series" not in values:
                        values["series"] = []
                    values["series"].append("Cisco Business FindIT")
                    concepts.append(values.copy())
        return concepts

    def save_videos_with_empty_series(
        self, path: str = f"{VIDEOS_DATA_DIR}/youtube_videos_no_series.json"
    ):
        try:
            with open(path, "w") as file:
                videos = [
                    video.model_dump()
                    for video in self.videos
                    if not video.series or len(video.series) == 0
                ]
                json.dump(videos, file, indent=2)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error saving videos with empty series to JSON file. Error: {e}")

    def save_videos_to_json(self, path: str):
        try:
            with open(path, "w") as file:
                videos = [video.model_dump() for video in self.videos]
                json.dump(videos, file, indent=2)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error saving videos to JSON file. Error: {e}")

    def save_videos_to_csv(self, path: str):
        try:
            df = pd.DataFrame(self.videos)
            df.to_csv(path, index=True)
        except Exception as e:
            print(f"Error saving videos to CSV file. Error: {e}")


if __name__ == "__main__":
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    PLAYLIST_ID = "PLB4F91009260AB3D7"
    CHANNEL_IDS = ["UCEWiIE6Htd8mvlOR6YQez1g"]

    youtube_loader = CiscoYouTubeDataLoader(
        GOOGLE_API_KEY, PLAYLIST_ID, CHANNEL_IDS, fetch=True
    )

    youtube_loader.save_videos_to_csv(f"{VIDEOS_DATA_DIR}/youtube_smbvideos.csv")
    youtube_loader.save_videos_to_json(f"{VIDEOS_DATA_DIR}/youtube_smbvideos.json")
    youtube_loader.save_videos_with_empty_series(
        f"{VIDEOS_DATA_DIR}/youtube_smbvideos_no_series.json"
    )
