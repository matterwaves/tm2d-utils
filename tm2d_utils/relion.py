import atexit
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import mrcfile
import starfile


def read_mrc(mrc_path):
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        return mrc.data.copy()


def read_starfile(star_fpath):
    return starfile.read(star_fpath)


class RelionWorkspace:
    def __init__(
        self,
        root_dir=None,
        root_is_remote=True,
        remote_host="frits.qb3.berkeley.edu",
        remote_user="spuser",
        port=22,
        key_filename=None,
    ):
        self.root_dir = root_dir
        self.root_is_remote = root_is_remote
        self.remote_host = remote_host
        self.remote_user = remote_user
        self.port = port
        self.key_filename = key_filename
        self._ssh = None
        self._sftp = None

    def _ensure_sftp(self):
        if not self.root_is_remote:
            return None

        if self._sftp is not None and self._ssh is not None:
            transport = self._ssh.get_transport()
            if transport is not None and transport.is_active():
                return self._sftp
            self.close_remote()

        try:
            import paramiko
        except ImportError as exc:
            raise ImportError(
                "Remote RELION workspaces require paramiko. Install paramiko or use "
                "--remote-is-true false for a locally mounted workspace."
            ) from exc

        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = {
            "hostname": self.remote_host,
            "port": self.port,
            "username": self.remote_user,
        }
        if self.key_filename is not None:
            connect_kwargs["key_filename"] = self.key_filename
        self._ssh.connect(**connect_kwargs)

        transport = self._ssh.get_transport()
        if transport is not None:
            transport.set_keepalive(30)

        self._sftp = self._ssh.open_sftp()
        return self._sftp

    def open_remote(self):
        self._ensure_sftp()
        if self.root_is_remote:
            atexit.register(self.close_remote)

    def close_remote(self):
        if self._sftp is not None:
            try:
                self._sftp.close()
            finally:
                self._sftp = None
        if self._ssh is not None:
            try:
                self._ssh.close()
            finally:
                self._ssh = None

    def fetch_remote_to_temp(self, remote_fpath, suffix=""):
        if not self.root_is_remote:
            return remote_fpath

        sftp = self._ensure_sftp()
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        local_tmp_fpath = tmp.name

        try:
            sftp.get(remote_fpath, local_tmp_fpath)
        except Exception:
            if os.path.exists(local_tmp_fpath):
                os.remove(local_tmp_fpath)
            raise

        return local_tmp_fpath

    def read_mrc_remote(self, mrc_fpath):
        if not self.root_is_remote:
            return read_mrc(mrc_fpath)

        local_tmp = self.fetch_remote_to_temp(mrc_fpath, suffix=".mrc")
        try:
            return read_mrc(local_tmp)
        finally:
            if os.path.exists(local_tmp):
                os.remove(local_tmp)

    def read_starfile(self, star_fpath):
        if not self.root_is_remote:
            return read_starfile(star_fpath)

        local_tmp = self.fetch_remote_to_temp(star_fpath, suffix=".star")
        try:
            return read_starfile(local_tmp)
        finally:
            if os.path.exists(local_tmp):
                os.remove(local_tmp)

    def get_job_dir(self, job_type, job_num):
        job_num = int(job_num)
        width = max(3, len(str(abs(job_num))))
        job_num_str = f"job{job_num:0{width}d}"
        if self.root_dir is None:
            return os.path.join(job_type, job_num_str)
        return os.path.join(self.root_dir, job_type, job_num_str)

    def __enter__(self):
        self.open_remote()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close_remote()
        return False


class RelionSession:
    def __init__(self, workspace: RelionWorkspace, name=None, laser=None, **job_numbers):
        self.workspace = workspace
        self.name = name
        self.laser = laser
        for job_type, job_num in job_numbers.items():
            setattr(self, job_type, job_num)


@dataclass
class SessionConfig:
    session: RelionSession
    job_type: str
    color: str = "tab:blue"


def make_session_config(
    workspace_root,
    workspace_root_is_remote,
    session_job_type,
    session_job_num,
    session_name=None,
    session_laser_state=None,
    color="tab:blue",
    **workspace_kwargs: Any,
):
    workspace = RelionWorkspace(
        root_dir=workspace_root,
        root_is_remote=workspace_root_is_remote,
        **workspace_kwargs,
    )
    session_kwargs = {session_job_type: session_job_num}
    if session_name not in {None, ""}:
        session_kwargs["name"] = session_name
    if session_laser_state is not None:
        session_kwargs["laser"] = session_laser_state

    session = RelionSession(workspace=workspace, **session_kwargs)
    return SessionConfig(session=session, job_type=session_job_type, color=color)
