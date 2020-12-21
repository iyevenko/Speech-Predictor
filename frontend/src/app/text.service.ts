import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class TextService {

  constructor(private http: HttpClient) { }

  submitText(input: any) {
    const headers = {text: input};
    const url = environment.apiUrl + '/rest/review-sentiment/';
    return this.http.get(url, {'headers': headers});
  }

}
